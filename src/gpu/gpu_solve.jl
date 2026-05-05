# gpu_solve.jl — Step 5
#
# Top-level entry point: cpu mesh data → GPU solve → cpu displacement vector.
#
# Includes (in dependency order):
#   device_functions.jl          — @inline math helpers for kernels
#   element_stiffness_kernel.jl  — GPU ke assembly (one thread per element)
#   assembly_kernel.jl           — GPU CSR scatter + CPU CSR pattern build
#   dirichlet_kernel.jl          — GPU BC application + CPU helper builders
#   pcg_solver.jl                — PCG kernels + cuBLAS + CUSOLVER backends
#
# ── Main function ─────────────────────────────────────────────────────────────
#
#   u, stats = gpu_solve(mesh_params, material, fixed_dofs, f;
#                        solver=:custom_pcg, tol=1e-8, max_iter=10_000)
#
# mesh_params NamedTuple fields:
#   n_elem        — number of elements                      (Int)
#   n_nodes       — number of nodes                         (Int)
#   n_dof         — total DOFs = n_nodes * 6                (Int)
#   node_coords   — (n_nodes × 3) Float64   global XYZ     (Matrix)
#   connectivity  — (n_elem  × 3) Int32     node indices   (Matrix)
#   dof_map       — (n_elem  × 18) Int32    global DOF ids  (Matrix)
#                   Built by Ferrite DOF handler (field-by-field ordering).
#   element_coords— (n_elem  × 3 × 3) Float64  XYZ per node of each element
#                   Precomputed from node_coords + connectivity.
#
# material NamedTuple fields:  E (Young's modulus), nu (Poisson ratio), t (thickness)
# fixed_dofs  : Vector{Int32}  — 1-based global DOF indices with zero displacement
# f           : Vector{Float64} — global force vector (length n_dof), CPU array
#
# Returns:
#   u     : Vector{Float64}  (length n_dof)  — displacement solution
#   stats : NamedTuple  (solver, iters, rel_residual, timings)
#           timings keys: cpu_preproc, upload, ke_kernel, assembly,
#                         dirichlet, precond, solve, download

using CUDA
using CUDA.CUBLAS

include("device_functions.jl")
include("element_stiffness_kernel.jl")
include("assembly_kernel.jl")
include("dirichlet_kernel.jl")
include("pcg_solver.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────
function gpu_solve(mesh_params,
                   material,
                   fixed_dofs ::Vector{Int32},
                   f          ::Vector{Float64};
                   solver     ::Symbol  = :custom_pcg,
                   tol        ::Float64 = 1e-8,
                   max_iter   ::Int     = 10_000)

    n_elem  = mesh_params.n_elem
    n_dof   = mesh_params.n_dof
    dof_map = mesh_params.dof_map           # Matrix{Int32} (n_elem × 18)
    coords  = mesh_params.element_coords    # Array{Float64,3} (n_elem × 3 × 3)

    E  = Float64(material.E)
    ν  = Float64(material.nu)
    t  = Float64(material.t)

    timings = Dict{Symbol,Float64}()

    # ── Phase 1: CPU preprocessing ────────────────────────────────────────────
    t0 = time_ns()

    rowptr, colind, scatter_idx = build_csr_pattern(dof_map, n_dof)
    nnz = length(colind)

    diag_idx          = build_diag_idx(rowptr, colind, n_dof)
    col_ptr, col_vals = build_col_index(rowptr, colind, fixed_dofs, n_dof)

    timings[:cpu_preproc] = (time_ns() - t0) * 1e-6

    # ── Phase 2: Upload to GPU ────────────────────────────────────────────────
    t0 = time_ns()

    coords_d      = CuArray(coords)                    # (n_elem,3,3) Float64
    scatter_idx_d = CuArray(scatter_idx)               # (n_elem,18,18) Int32
    rowptr_d      = CuArray(rowptr)                    # (n_dof+1,) Int32
    colind_d      = CuArray(colind)                    # (nnz,) Int32
    diag_idx_d    = CuArray(diag_idx)                  # (n_dof,) Int32
    col_ptr_d     = CuArray(col_ptr)                   # (n_fixed+1,) Int32
    col_vals_d    = CuArray(col_vals)                  # (total_col_entries,) Int32
    fixed_dofs_d  = CuArray(fixed_dofs)                # (n_fixed,) Int32
    K_values_d    = CUDA.zeros(Float64, nnz)           # zero-init CSR values
    f_d           = CuArray(f)                          # (n_dof,) Float64
    CUDA.synchronize()

    timings[:upload] = (time_ns() - t0) * 1e-6

    # ── Phase 3: Element stiffness kernel ─────────────────────────────────────
    t0 = time_ns()

    ke_all_d   = CUDA.zeros(Float64, n_elem, 18, 18)
    threads_ke = 256
    blocks_ke  = cld(n_elem, threads_ke)
    @cuda threads=threads_ke blocks=blocks_ke element_stiffness_kernel!(
        ke_all_d, coords_d, E, ν, t, Int32(n_elem))
    CUDA.synchronize()

    timings[:ke_kernel] = (time_ns() - t0) * 1e-6

    # ── Phase 4: Assembly kernel ──────────────────────────────────────────────
    t0 = time_ns()

    threads_asm = 256
    blocks_asm  = cld(n_elem, threads_asm)
    @cuda threads=threads_asm blocks=blocks_asm assembly_kernel!(
        K_values_d, ke_all_d, scatter_idx_d, Int32(n_elem))
    CUDA.synchronize()

    # ke_all no longer needed — free GPU memory
    ke_all_d = nothing
    GC.gc(false)

    timings[:assembly] = (time_ns() - t0) * 1e-6

    # ── Phase 5: Dirichlet BCs ────────────────────────────────────────────────
    t0 = time_ns()

    n_fixed     = Int32(length(fixed_dofs))
    threads_dir = min(n_fixed, Int32(256))
    blocks_dir  = cld(Int(n_fixed), Int(threads_dir))

    @cuda threads=threads_dir blocks=blocks_dir dirichlet_row_kernel!(
        K_values_d, f_d, rowptr_d, diag_idx_d, fixed_dofs_d, n_fixed)
    CUDA.synchronize()

    if n_fixed > 0
        # One block per fixed DOF, 256 threads per block
        @cuda threads=256 blocks=Int(n_fixed) dirichlet_col_kernel!(
            K_values_d, col_ptr_d, col_vals_d, n_fixed)
        CUDA.synchronize()
    end

    timings[:dirichlet] = (time_ns() - t0) * 1e-6

    # ── Phase 6: Jacobi preconditioner ────────────────────────────────────────
    t0 = time_ns()

    M_inv_d = CUDA.zeros(Float64, n_dof)
    build_jacobi_precond!(M_inv_d, K_values_d, rowptr_d, colind_d, n_dof)
    CUDA.synchronize()

    timings[:precond] = (time_ns() - t0) * 1e-6

    # ── Phase 7: Solve ────────────────────────────────────────────────────────
    t0 = time_ns()

    u_d, solve_stats = solve_gpu(rowptr_d, colind_d, K_values_d, f_d, M_inv_d;
                                  backend=solver, tol=tol, max_iter=max_iter)
    CUDA.synchronize()

    timings[:solve] = (time_ns() - t0) * 1e-6

    # ── Phase 8: Download ─────────────────────────────────────────────────────
    t0 = time_ns()
    u  = Array(u_d)
    timings[:download] = (time_ns() - t0) * 1e-6

    stats = (
        solver       = solve_stats.solver,
        iters        = solve_stats.iters,
        rel_residual = solve_stats.rel_residual,
        timings      = timings,
    )

    return u, stats
end

# ─────────────────────────────────────────────────────────────────────────────
# Buckling extension stub
# ─────────────────────────────────────────────────────────────────────────────
# See gpu_solve.jl design doc (Step B–E) for the full pipeline.
# Requires geometric_stiffness_kernel.jl to be included first.
#
# NOTE: CPU reference bug in assemble_global_Kg! lines 605-607:
#   σxx_element is referenced but never assigned from σXX_element.
#   Fix: use σXX[i], σYY[i], τXY[i] directly.
function gpu_solve_buckling(mesh_params,
                             material,
                             u          ::Vector{Float64},
                             fixed_dofs ::Vector{Int32};
                             n_modes    ::Int = 5)
    error("gpu_solve_buckling not yet implemented — include geometric_stiffness_kernel.jl first")
end
