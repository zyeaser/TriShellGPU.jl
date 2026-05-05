# dirichlet_kernel.jl — Step 3b
#
# GPU-resident Dirichlet BC application.
# Must run AFTER assembly_kernel! and BEFORE build_jacobi_precond!.
#
# Two GPU kernels (run in order, with synchronize between them):
#   dirichlet_row_kernel!  — zero rows, set diag=1, set f[g]=0
#   dirichlet_col_kernel!  — zero columns (preserves symmetry)
#
# Two CPU helpers (run once in build phase alongside build_csr_pattern):
#   build_diag_idx         — finds diagonal entry flat CSR index per DOF
#   build_col_index        — builds ragged column-entry lists for fixed DOFs
#
# ── Usage (inside gpu_solve.jl) ───────────────────────────────────────────────
#
#   diag_idx             = build_diag_idx(rowptr, colind, n_dof)
#   col_ptr, col_vals    = build_col_index(rowptr, colind, fixed_dofs, n_dof)
#
#   diag_idx_d   = CuArray(diag_idx)
#   col_ptr_d    = CuArray(col_ptr)
#   col_vals_d   = CuArray(col_vals)
#   fixed_dofs_d = CuArray(fixed_dofs)
#   n_fixed      = Int32(length(fixed_dofs))
#
#   threads_r = min(n_fixed, Int32(256))
#   blocks_r  = cld(n_fixed, threads_r)
#   @cuda threads=threads_r blocks=blocks_r dirichlet_row_kernel!(
#       K_values_d, f_d, rowptr_d, diag_idx_d, fixed_dofs_d, n_fixed)
#   CUDA.synchronize()
#
#   @cuda threads=256 blocks=Int32(n_fixed) dirichlet_col_kernel!(
#       K_values_d, col_ptr_d, col_vals_d, n_fixed)

using CUDA

# ─────────────────────────────────────────────────────────────────────────────
# CPU helper: diagonal index per DOF
# ─────────────────────────────────────────────────────────────────────────────
# Returns diag_idx::Vector{Int32} where diag_idx[g] is the flat 1-based CSR
# index k such that colind[k] == g in row g of the CSR matrix.
function build_diag_idx(rowptr::Vector{Int32}, colind::Vector{Int32}, n_dof::Int)
    diag_idx = Vector{Int32}(undef, n_dof)

    @inbounds for g in 1:n_dof
        rstart = Int(rowptr[g])
        rend   = Int(rowptr[g + 1]) - 1
        lo, hi = rstart, rend
        found  = Int32(0)
        while lo <= hi
            mid = (lo + hi) >>> 1
            cv  = colind[mid]
            if cv < g
                lo = mid + 1
            elseif cv > g
                hi = mid - 1
            else
                found = Int32(mid)
                break
            end
        end
        diag_idx[g] = found
    end

    return diag_idx
end

# ─────────────────────────────────────────────────────────────────────────────
# CPU helper: ragged column-entry lists for fixed DOFs
# ─────────────────────────────────────────────────────────────────────────────
# Returns (col_ptr, col_vals):
#   col_ptr  : Vector{Int32}  length n_fixed+1  — 1-based CSR-style pointers
#   col_vals : Vector{Int32}  — flat array of CSR flat-indices k where
#              colind[k] == fixed_dofs[i], for each i in 1:n_fixed
#
# fixed_dofs : Vector{Int32}  — 1-based global DOF indices that are prescribed
function build_col_index(rowptr    ::Vector{Int32},
                         colind    ::Vector{Int32},
                         fixed_dofs::Vector{Int32},
                         n_dof     ::Int)
    n_fixed = length(fixed_dofs)

    # Map from DOF index → position in fixed_dofs (for fast lookup)
    fixed_pos = Dict{Int32,Int}()
    sizehint!(fixed_pos, n_fixed)
    for (i, g) in enumerate(fixed_dofs)
        fixed_pos[g] = i
    end

    # Build row_of_entry: for each flat CSR index k, which row it belongs to
    nnz = length(colind)
    row_of_entry = Vector{Int32}(undef, nnz)
    for r in 1:n_dof
        for k in Int(rowptr[r]) : Int(rowptr[r+1]) - 1
            row_of_entry[k] = Int32(r)
        end
    end

    # Accumulate off-diagonal entries per fixed DOF column.
    # The diagonal K[g,g] is excluded so dirichlet_row_kernel!'s K[g,g]=1 is preserved.
    col_entries = [Vector{Int32}() for _ in 1:n_fixed]

    @inbounds for k in 1:nnz
        g = colind[k]
        row_of_entry[k] == g && continue   # skip diagonal
        pos = get(fixed_pos, g, 0)
        if pos > 0
            push!(col_entries[pos], Int32(k))
        end
    end

    # Pack into CSR-style (col_ptr, col_vals)
    col_ptr = Vector{Int32}(undef, n_fixed + 1)
    col_ptr[1] = Int32(1)
    for i in 1:n_fixed
        col_ptr[i + 1] = col_ptr[i] + Int32(length(col_entries[i]))
    end

    total    = Int(col_ptr[n_fixed + 1]) - 1
    col_vals = Vector{Int32}(undef, total)
    for i in 1:n_fixed
        base = Int(col_ptr[i]) - 1
        for (j, k) in enumerate(col_entries[i])
            col_vals[base + j] = k
        end
    end

    return col_ptr, col_vals
end

# ─────────────────────────────────────────────────────────────────────────────
# GPU kernel 1: zero rows, set diagonal = 1, zero f[g]
# ─────────────────────────────────────────────────────────────────────────────
# One thread per fixed DOF.
function dirichlet_row_kernel!(
        K_values  ::CuDeviceArray{Float64,1},
        f         ::CuDeviceArray{Float64,1},
        rowptr    ::CuDeviceArray{Int32,1},
        diag_idx  ::CuDeviceArray{Int32,1},
        fixed_dofs::CuDeviceArray{Int32,1},
        n_fixed   ::Int32)

    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    tid > n_fixed && return

    @inbounds begin
        g      = fixed_dofs[tid]
        rstart = rowptr[g]
        rend   = rowptr[g + Int32(1)] - Int32(1)

        # Zero entire row
        for k in rstart:rend
            K_values[k] = 0.0
        end

        # Set diagonal and RHS
        K_values[diag_idx[g]] = 1.0
        f[g]                  = 0.0
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# GPU kernel 2: zero columns for all fixed DOFs
# ─────────────────────────────────────────────────────────────────────────────
# Launch as 1D grid: one block per fixed DOF, 256 threads per block.
# blockIdx().x (1-based) = i  — index into fixed_dofs
# threadIdx().x loops over entries in col_vals[col_ptr[i]:col_ptr[i+1]-1]
#
# NOTE: dirichlet_row_kernel! must complete (CUDA.synchronize()) before this
#       runs, so the diagonal 1.0 set in kernel 1 is not overwritten here.
function dirichlet_col_kernel!(
        K_values::CuDeviceArray{Float64,1},
        col_ptr ::CuDeviceArray{Int32,1},
        col_vals::CuDeviceArray{Int32,1},
        n_fixed ::Int32)

    i = blockIdx().x
    i > n_fixed && return

    @inbounds begin
        rstart = col_ptr[i]
        rend   = col_ptr[i + Int32(1)] - Int32(1)
        len    = rend - rstart + Int32(1)

        # Stride loop: one block-worth of threads covers all entries
        j = threadIdx().x
        while j <= len
            K_values[col_vals[rstart + j - Int32(1)]] = 0.0
            j += blockDim().x
        end
    end

    return nothing
end
