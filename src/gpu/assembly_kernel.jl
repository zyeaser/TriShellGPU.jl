# assembly_kernel.jl — Step 3
#
# Two parts:
#   1. build_csr_pattern  — CPU, runs once before any GPU launch
#   2. assembly_kernel!   — GPU, scatters ke_all into CSR K_values via atomic adds
#
# Dependencies: none beyond Base Julia and CUDA.
#
# ── CPU: build_csr_pattern ────────────────────────────────────────────────────
#
#   rowptr, colind, scatter_idx = build_csr_pattern(dof_map, n_dof)
#
#   dof_map  : Matrix{Int32}  (n_elem × 18)  — global DOF indices per element
#              1-based, Ferrite field-by-field ordering (already matches ke_all)
#   n_dof    : Int            — total DOFs
#
#   rowptr   : Vector{Int32}  length n_dof+1  — 1-based CSR row pointers
#   colind   : Vector{Int32}  length nnz      — column indices (sorted per row)
#   scatter_idx : Array{Int32,3}  (n_elem,18,18)
#              scatter_idx[e,i,j] = flat 1-based CSR index k such that
#              colind[k] == dof_map[e,j] in the row dof_map[e,i]
#              Enables direct indexed write in assembly_kernel! (no GPU search).
#
# ── GPU: assembly_kernel! ─────────────────────────────────────────────────────
#
#   One thread per element.
#   For each (i,j) in 1:18 × 1:18:
#       K_values[scatter_idx[e,i,j]] += ke_all[e,i,j]   (atomic)
#
#   K_values must be CUDA.zeros(Float64, nnz) before launch.
#
# ── Usage (inside gpu_solve.jl) ───────────────────────────────────────────────
#
#   rowptr, colind, scatter_idx = build_csr_pattern(dof_map, n_dof)
#   scatter_idx_d = CuArray(scatter_idx)
#   K_values_d    = CUDA.zeros(Float64, length(colind))
#   threads = 256; blocks = cld(n_elem, threads)
#   @cuda threads=threads blocks=blocks assembly_kernel!(
#       K_values_d, ke_all_d, scatter_idx_d, Int32(n_elem))

using CUDA

# ─────────────────────────────────────────────────────────────────────────────
# CPU helper: build CSR pattern + scatter index
# ─────────────────────────────────────────────────────────────────────────────
function build_csr_pattern(dof_map::Matrix{Int32}, n_dof::Int)
    n_elem = size(dof_map, 1)

    # ── 1. Collect all (row, col) pairs ──────────────────────────────────────
    pairs = Vector{Tuple{Int32,Int32}}()
    sizehint!(pairs, n_elem * 324)   # 18×18 upper bound

    @inbounds for e in 1:n_elem
        for i in 1:18
            gi = dof_map[e, i]
            for j in 1:18
                gj = dof_map[e, j]
                push!(pairs, (gi, gj))
            end
        end
    end

    sort!(pairs)
    unique!(pairs)
    nnz = length(pairs)

    # ── 2. Build rowptr and colind ────────────────────────────────────────────
    # rowptr[i] = 1-based start of row i; rowptr[i+1]-1 = end of row i
    rowptr = zeros(Int32, n_dof + 1)
    colind = Vector{Int32}(undef, nnz)

    @inbounds for k in 1:nnz
        r, c = pairs[k]
        rowptr[r + 1] += Int32(1)
        colind[k]      = c
    end

    rowptr[1] = Int32(1)
    @inbounds for i in 1:n_dof
        rowptr[i + 1] += rowptr[i]
    end

    # ── 3. Build scatter_idx via binary search ────────────────────────────────
    scatter_idx = Array{Int32,3}(undef, n_elem, 18, 18)

    @inbounds for e in 1:n_elem
        for i in 1:18
            gi     = dof_map[e, i]
            rstart = Int(rowptr[gi])
            rend   = Int(rowptr[gi + 1]) - 1

            for j in 1:18
                gj = dof_map[e, j]
                # Binary search: colind is sorted within [rstart, rend]
                lo, hi = rstart, rend
                found = Int32(0)
                while lo <= hi
                    mid = (lo + hi) >>> 1
                    cv  = colind[mid]
                    if cv < gj
                        lo = mid + 1
                    elseif cv > gj
                        hi = mid - 1
                    else
                        found = Int32(mid)
                        break
                    end
                end
                scatter_idx[e, i, j] = found
            end
        end
    end

    return rowptr, colind, scatter_idx
end

# ─────────────────────────────────────────────────────────────────────────────
# GPU kernel: atomic scatter of ke_all into K_values
# ─────────────────────────────────────────────────────────────────────────────
function assembly_kernel!(
        K_values   ::CuDeviceArray{Float64,1},
        ke_all     ::CuDeviceArray{Float64,3},
        scatter_idx::CuDeviceArray{Int32,3},
        n_elem     ::Int32)

    e = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    e > n_elem && return

    @inbounds for i in 1:18
        for j in 1:18
            val = ke_all[e, i, j]
            idx = scatter_idx[e, i, j]
            CUDA.atomic_add!(pointer(K_values, idx), val)
        end
    end

    return nothing
end
