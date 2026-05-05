# pcg_solver.jl — Step 4
#
# Three solver backends for K * u = f (all operating on GPU CSR arrays):
#   Solver A — custom CUDA PCG kernels  (default)
#   Solver B — cuBLAS dot/axpy, fewer CPU-GPU scalar transfers
#   Solver C — CUSOLVER sparse direct (small meshes only)
#
# Shared entry point:
#   u, stats = solve_gpu(rowptr, colind, K_values, f, M_inv;
#                        backend=:custom_pcg, tol=1e-8, max_iter=10_000)
#
# All CuArray inputs are NOT copied — K_values, f must be ready after
# assembly_kernel! and dirichlet_kernel!.

using CUDA
using CUDA.CUBLAS
using CUDA.CUSPARSE

# ─────────────────────────────────────────────────────────────────────────────
# Shared: Jacobi preconditioner build
# ─────────────────────────────────────────────────────────────────────────────
# One thread per DOF.  Scans CSR row for the diagonal entry and writes
# M_inv[i] = 1 / K[i,i].  After dirichlet_kernel!, fixed DOFs have K[g,g]=1
# so M_inv[g]=1 — PCG correctly treats them as already converged.
function build_jacobi_precond_kernel!(
        M_inv    ::CuDeviceArray{Float64,1},
        K_values ::CuDeviceArray{Float64,1},
        rowptr   ::CuDeviceArray{Int32,1},
        colind   ::CuDeviceArray{Int32,1},
        n        ::Int32)

    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return

    @inbounds begin
        rstart = rowptr[i]
        rend   = rowptr[i + Int32(1)] - Int32(1)
        diag   = 1.0
        for k in rstart:rend
            if colind[k] == i
                diag = K_values[k]
                break
            end
        end
        M_inv[i] = 1.0 / diag
    end

    return nothing
end

# Host wrapper
function build_jacobi_precond!(M_inv, K_values, rowptr_d, colind_d, n_dof)
    threads = 256
    blocks  = cld(n_dof, threads)
    @cuda threads=threads blocks=blocks build_jacobi_precond_kernel!(
        M_inv, K_values, rowptr_d, colind_d, Int32(n_dof))
end

# ─────────────────────────────────────────────────────────────────────────────
# Solver A kernels — custom CUDA
# ─────────────────────────────────────────────────────────────────────────────
const FULL_MASK = UInt32(0xffffffff)

# y = K * x  (CSR SpMV, one thread per row)
function spmv_kernel!(
        y      ::CuDeviceArray{Float64,1},
        rowptr ::CuDeviceArray{Int32,1},
        colind ::CuDeviceArray{Int32,1},
        values ::CuDeviceArray{Float64,1},
        x      ::CuDeviceArray{Float64,1},
        n      ::Int32)

    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return

    @inbounds begin
        s = 0.0
        for k in rowptr[i] : rowptr[i + Int32(1)] - Int32(1)
            s += values[k] * x[colind[k]]
        end
        y[i] = s
    end

    return nothing
end

# result += dot(a, b)  (parallel reduction, result must be zeroed before call)
# Uses 256-thread blocks and a grid-stride loop so any n is handled.
function dot_kernel!(
        result::CuDeviceArray{Float64,1},
        a     ::CuDeviceArray{Float64,1},
        b     ::CuDeviceArray{Float64,1},
        n     ::Int32)

    tid  = threadIdx().x
    bid  = blockIdx().x
    bdim = blockDim().x
    gdim = gridDim().x

    shared = CuStaticSharedArray(Float64, 256)

    # Grid-stride accumulate
    i = (bid - Int32(1)) * bdim + tid
    s = 0.0
    while i <= n
        @inbounds s += a[i] * b[i]
        i += gdim * bdim
    end
    @inbounds shared[tid] = s
    sync_threads()

    # Tree reduction within the block (power-of-2 steps down to 1)
    stride = bdim >>> 1
    while stride > 0
        if tid <= stride
            @inbounds shared[tid] += shared[tid + stride]
        end
        sync_threads()
        stride >>>= 1
    end

    if tid == Int32(1)
        @inbounds CUDA.atomic_add!(pointer(result, 1), shared[1])
    end

    return nothing
end

# result += dot(a, b)  — tiled variant
# T=4 unrolled grid-stride loop + warp-shuffle reduction.
# Shared memory: 8 slots (one per warp) instead of 256.
# Sync barriers: 1 instead of 8.
# Launch config identical to dot_kernel!: threads=256, blocks=cld(n,256).
function dot_kernel_tiled!(
        result::CuDeviceArray{Float64,1},
        a     ::CuDeviceArray{Float64,1},
        b     ::CuDeviceArray{Float64,1},
        n     ::Int32)

    tid  = threadIdx().x
    bid  = blockIdx().x
    bdim = blockDim().x
    gdim = gridDim().x

    shared = CuStaticSharedArray(Float64, 8)   # one slot per warp

    # ── Phase 1: T=4 unrolled grid-stride accumulate ─────────────────────────
    i       = (bid - Int32(1)) * bdim + tid
    stride1 = gdim * bdim
    stride4 = Int32(4) * stride1
    s       = 0.0

    while i + Int32(3) * stride1 <= n
        @inbounds s += a[i]                      * b[i]
        @inbounds s += a[i + stride1]             * b[i + stride1]
        @inbounds s += a[i + Int32(2) * stride1]  * b[i + Int32(2) * stride1]
        @inbounds s += a[i + Int32(3) * stride1]  * b[i + Int32(3) * stride1]
        i += stride4
    end
    while i <= n                                 # scalar cleanup for remainder
        @inbounds s += a[i] * b[i]
        i += stride1
    end

    # ── Phase 2: Intra-warp reduction (5 shuffles, 0 sync_threads) ──────────
    s += CUDA.shfl_down_sync(FULL_MASK, s, 16)
    s += CUDA.shfl_down_sync(FULL_MASK, s,  8)
    s += CUDA.shfl_down_sync(FULL_MASK, s,  4)
    s += CUDA.shfl_down_sync(FULL_MASK, s,  2)
    s += CUDA.shfl_down_sync(FULL_MASK, s,  1)

    # ── Phase 3: Write 8 warp sums to shared memory ──────────────────────────
    # warp_id ∈ [1,8] (1-based);  lane_id ∈ [0,31] (0-based)
    warp_id = (tid - Int32(1)) >>> 5 + Int32(1)
    lane_id = (tid - Int32(1)) & Int32(31)

    if lane_id == Int32(0)
        @inbounds shared[warp_id] = s
    end
    sync_threads()                               # only barrier in the kernel

    # ── Phase 4: Final 8-element reduction in warp 1 (3 shuffles) ───────────
    if warp_id == Int32(1)
        s2 = tid <= Int32(8) ? (@inbounds shared[tid]) : 0.0
        s2 += CUDA.shfl_down_sync(FULL_MASK, s2, 4)
        s2 += CUDA.shfl_down_sync(FULL_MASK, s2, 2)
        s2 += CUDA.shfl_down_sync(FULL_MASK, s2, 1)
        if tid == Int32(1)
            @inbounds CUDA.atomic_add!(pointer(result, 1), s2)
        end
    end

    return nothing
end

# y[i] += alpha * x[i]
function axpy_kernel!(
        y    ::CuDeviceArray{Float64,1},
        alpha::Float64,
        x    ::CuDeviceArray{Float64,1},
        n    ::Int32)

    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    @inbounds y[i] += alpha * x[i]
    return nothing
end

# y[i] = alpha * x[i] + beta * z[i]
# Used for PCG direction update: p_new = z + beta * p_old  (y=p, x=z, z=p)
function axpby_kernel!(
        y    ::CuDeviceArray{Float64,1},
        alpha::Float64,
        x    ::CuDeviceArray{Float64,1},
        beta ::Float64,
        z    ::CuDeviceArray{Float64,1},
        n    ::Int32)

    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    @inbounds y[i] = alpha * x[i] + beta * z[i]
    return nothing
end

# u[i] += alpha*p[i];  r[i] -= alpha*Ap[i];  z[i] = M_inv[i]*r[i]
# Fuses all three per-iteration element-wise updates into one kernel launch:
# u-update, r-update, and Jacobi preconditioner applied to the new r.
function pcg_update_kernel!(
        u    ::CuDeviceArray{Float64,1},
        r    ::CuDeviceArray{Float64,1},
        z    ::CuDeviceArray{Float64,1},
        alpha::Float64,
        p    ::CuDeviceArray{Float64,1},
        Ap   ::CuDeviceArray{Float64,1},
        M_inv::CuDeviceArray{Float64,1},
        n    ::Int32)

    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    @inbounds u[i] += alpha * p[i]
    @inbounds r[i] -= alpha * Ap[i]
    @inbounds z[i]  = M_inv[i] * r[i]
    return nothing
end

# y[i] = alpha[i] * x[i]  (element-wise; used for z = M_inv .* r)
function scale_kernel!(
        y    ::CuDeviceArray{Float64,1},
        alpha::CuDeviceArray{Float64,1},
        x    ::CuDeviceArray{Float64,1},
        n    ::Int32)

    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    @inbounds y[i] = alpha[i] * x[i]
    return nothing
end

# ── Solver A host function ────────────────────────────────────────────────────
function pcg_solve_A(rowptr   ::CuArray{Int32,1},
                     colind   ::CuArray{Int32,1},
                     K_values ::CuArray{Float64,1},
                     f        ::CuArray{Float64,1},
                     M_inv    ::CuArray{Float64,1};
                     tol      ::Float64 = 1e-8,
                     max_iter ::Int     = 10_000)

    n = Int32(length(f))
    threads = 256
    blocks  = cld(Int(n), threads)

    K_csr = CuSparseMatrixCSR{Float64, Int32}(rowptr, colind, K_values, (n, n))

    u  = CUDA.zeros(Float64, n)
    r  = copy(f)
    z  = CUDA.zeros(Float64, n)
    p  = CUDA.zeros(Float64, n)
    Ap = CUDA.zeros(Float64, n)

    # gpu_dot(a, b) = CUBLAS.dot(n, a, b)
    buf = CUDA.zeros(Float64, 1)
    function gpu_dot(a, b)
        fill!(buf, 0.0)
        @cuda threads=threads blocks=blocks dot_kernel_tiled!(buf, a, b, n)
        return Array(buf)[1]
    end

    # Initial z = M_inv .* r,  p = z
    @cuda threads=threads blocks=blocks scale_kernel!(z, M_inv, r, n)
    copyto!(p, z)
    rz = gpu_dot(r, z)

    norm_f = sqrt(gpu_dot(f, f))
    norm_f = max(norm_f, 1e-300)

    t_start = time_ns()
    iters   = 0
    rel_res = NaN

    for iter in 1:max_iter
        CUSPARSE.mv!('N', 1.0, K_csr, p, 0.0, Ap, 'O')

        pAp   = gpu_dot(p, Ap)
        alpha = rz / pAp

        # FUSED: u += alpha*p,  r -= alpha*Ap,  z = M_inv.*r — one kernel launch.
        @cuda threads=threads blocks=blocks pcg_update_kernel!(u, r, z, alpha, p, Ap, M_inv, n)

        r_norm  = sqrt(gpu_dot(r, r))
        rel_res = r_norm / norm_f

        if rel_res < tol
            iters = iter
            break
        end

        iter % 100 == 0 && @info "PCG-A iter=$iter  rel_res=$rel_res"

        rz_new = gpu_dot(r, z)
        beta   = rz_new / rz

        @cuda threads=threads blocks=blocks axpby_kernel!(p, 1.0, z, beta, p, n)
        rz = rz_new
    end

    if iters == 0
        iters   = max_iter
        @warn "PCG-A did not converge within $max_iter iterations (rel_res=$rel_res)"
    end

    time_ms = (time_ns() - t_start) * 1e-6
    stats   = (solver=:custom_pcg, iters=iters, rel_residual=rel_res, time_ms=time_ms)
    return u, stats
end

# ─────────────────────────────────────────────────────────────────────────────
# Solver B — cuBLAS PCG
# ─────────────────────────────────────────────────────────────────────────────
# Same PCG algorithm as Solver A; replaces custom dot/axpy/scale with cuBLAS.
# SpMV still uses custom spmv_kernel! (cuBLAS has no sparse SpMV).
# alpha, beta, rz are length-1 CuArrays so scalars stay on device.
# Only ONE Array() transfer per iteration (convergence check on the r_norm).
function pcg_solve_B(rowptr   ::CuArray{Int32,1},
                     colind   ::CuArray{Int32,1},
                     K_values ::CuArray{Float64,1},
                     f        ::CuArray{Float64,1},
                     M_inv    ::CuArray{Float64,1};
                     tol      ::Float64 = 1e-8,
                     max_iter ::Int     = 10_000)

    n       = Int32(length(f))
    threads = 256
    blocks  = cld(Int(n), threads)


    # Build cuSPARSE matrix descriptor once (wraps existing device arrays,
    # no copy).  Index base 'O' = one-based (Julia convention).
    K_csr = CuSparseMatrixCSR{Float64, Int32}(rowptr, colind, K_values, (n, n))

    u  = CUDA.zeros(Float64, n)
    r  = copy(f)
    z  = similar(f)
    p  = similar(f)
    Ap = similar(f)

    # Apply preconditioner: z = M_inv .* r  (broadcast, stays on device)
    z .= M_inv .* r
    copyto!(p, z)

    # Scalars on device (length-1 CuArrays)
    rz     = CUDA.fill(CUBLAS.dot(n, r, z), 1)
    norm_f = sqrt(CUBLAS.dot(n, f, f))
    norm_f = max(norm_f, 1e-300)

    t_start = time_ns()
    iters   = 0
    rel_res = NaN

    for iter in 1:max_iter
        # Ap = K * p
        # @cuda threads=threads blocks=blocks spmv_kernel!(Ap, rowptr, colind, K_values, p, n)
        CUSPARSE.mv!('N', 1.0, K_csr, p, 0.0, Ap, 'O')

        pAp   = CUBLAS.dot(n, p, Ap)     # scalar on CPU (CUBLAS.dot returns host val)
        alpha = Array(rz)[1] / pAp        # one transfer: rz scalar

        CUBLAS.axpy!(n,  alpha, p,  u)    # u += alpha*p
        CUBLAS.axpy!(n, -alpha, Ap, r)    # r -= alpha*Ap

        r_norm  = sqrt(CUBLAS.dot(n, r, r))
        rel_res = r_norm / norm_f

        if rel_res < tol
            iters = iter
            break
        end

        if iter % 100 == 0
            @info "PCG-B iter=$iter  rel_res=$(rel_res)"
        end

        z .= M_inv .* r
        rz_new = CUBLAS.dot(n, r, z)
        beta   = rz_new / Array(rz)[1]

        # p = z + beta * p  (CUBLAS.axpby!: y = alpha*x + beta*y)
        CUBLAS.axpby!(n, 1.0, z, beta, p)  # p = 1.0*z + beta*p
        fill!(rz, rz_new)
    end

    if iters == 0
        iters   = max_iter
        @warn "PCG-B did not converge within $max_iter iterations (rel_res=$rel_res)"
    end

    time_ms = (time_ns() - t_start) * 1e-6
    stats   = (solver=:cublas_pcg, iters=iters, rel_residual=rel_res, time_ms=time_ms)
    return u, stats
end

# ─────────────────────────────────────────────────────────────────────────────
# Solver C — CUSOLVER sparse direct
# ─────────────────────────────────────────────────────────────────────────────
# Thin wrapper around CUSOLVER sparse Cholesky / QR factorization.
# K must be symmetric positive definite (it is, after BC application).
# Practical limit: ~500k DOF (fill-in grows O(n^1.5) for 2-D shell meshes).
function cusolver_solve(rowptr  ::CuArray{Int32,1},
                        colind  ::CuArray{Int32,1},
                        K_values::CuArray{Float64,1},
                        f       ::CuArray{Float64,1})

    n   = Int32(length(f))
    nnz = Int32(length(K_values))

    t_start = time_ns()
    x = CUDA.zeros(Float64, n)

    # Convert 1-based CSR to 0-based for CUSOLVER
    rowptr0 = rowptr .- Int32(1)
    colind0 = colind .- Int32(1)

    # Try sparse Cholesky with AMD reordering (reorder=1)
    # CUSOLVER.csrlsvchol expects: m, nnz, descr, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x
    try
        CUDA.CUSOLVER.csrlsvchol!(x, rowptr0, colind0, K_values, f,
                                   Float64(1e-14), Int32(1))
    catch e
        @warn "CUSOLVER Cholesky failed ($e); falling back to sparse QR"
        CUDA.CUSOLVER.csrlsvqr!(x, rowptr0, colind0, K_values, f,
                                 Float64(1e-14), Int32(1))
    end

    CUDA.synchronize()
    time_ms = (time_ns() - t_start) * 1e-6
    stats   = (solver=:cusolver, iters=nothing, rel_residual=nothing, time_ms=time_ms)
    return x, stats
end

# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatch
# ─────────────────────────────────────────────────────────────────────────────
#
#   u, stats = solve_gpu(rowptr, colind, K_values, f, M_inv;
#                        backend=:custom_pcg, tol=1e-8, max_iter=10_000)
#
#   backend : :custom_pcg | :cublas_pcg | :cusolver
#   stats   : NamedTuple with (solver, iters, rel_residual, time_ms)
function solve_gpu(rowptr   ::CuArray{Int32,1},
                   colind   ::CuArray{Int32,1},
                   K_values ::CuArray{Float64,1},
                   f        ::CuArray{Float64,1},
                   M_inv    ::CuArray{Float64,1};
                   backend  ::Symbol  = :custom_pcg,
                   tol      ::Float64 = 1e-8,
                   max_iter ::Int     = 10_000)

    if backend === :custom_pcg
        return pcg_solve_A(rowptr, colind, K_values, f, M_inv;
                           tol=tol, max_iter=max_iter)
    elseif backend === :cublas_pcg
        return pcg_solve_B(rowptr, colind, K_values, f, M_inv;
                           tol=tol, max_iter=max_iter)
    elseif backend === :cusolver
        return cusolver_solve(rowptr, colind, K_values, f)
    else
        error("Unknown solver backend: $backend. Choose :custom_pcg, :cublas_pcg, or :cusolver")
    end
end
