using CUDA
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# INFRASTRUCTURE SMOKE TEST — CUDA.jl + kernel launch sanity check
#
# PURPOSE: Verify that CUDA.jl is functional on this machine/cluster and that
#          basic kernel launch, memory allocation, and CPU/GPU comparison work.
#          This is NOT a test of the actual 3D shell element kernel.
#
# WHAT IT TESTS: 2D Jacobian determinant (x,y only — no z, no local frame,
#                no IP3/IP6, no rotation matrix). The 2D detJ formula is a
#                simple standalone computation chosen only because it is easy
#                to verify by hand.
#
# WHAT IT DOES NOT TEST:
#   - 3D coordinate rotation (local_to_global_rotation, expand_rotation_18)
#   - Shape functions (shape_functions_ip3, shape_functions_ip6)
#   - B-matrix assembly (B_membrane, B_bending, B_shear)
#   - Static condensation, drilling DOF penalty, Ferrite DOF reordering
#   For those, see: experiments/test_single_element.jl
#
# ─────────────────────────────────────────────────────────────────────────────

# ── GPU kernel: one thread per element ───────────────────────────────────────
function jacobian_kernel!(detJ_out, coords)
    e = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    e > size(coords, 2) && return

    # Node coordinates for element e
    x1 = coords[1, e];  y1 = coords[2, e]
    x2 = coords[3, e];  y2 = coords[4, e]
    x3 = coords[5, e];  y3 = coords[6, e]

    # 2x2 Jacobian
    J11 = x2 - x1;  J12 = y2 - y1
    J21 = x3 - x1;  J22 = y3 - y1

    detJ_out[e] = J11 * J22 - J12 * J21

    return
end

# ── CPU reference ─────────────────────────────────────────────────────────────
function compute_detJ_cpu!(detJ_out, coords)
    for e in axes(coords, 2)
        x1 = coords[1, e];  y1 = coords[2, e]
        x2 = coords[3, e];  y2 = coords[4, e]
        x3 = coords[5, e];  y3 = coords[6, e]

        J11 = x2 - x1;  J12 = y2 - y1
        J21 = x3 - x1;  J22 = y3 - y1

        detJ_out[e] = J11 * J22 - J12 * J21
    end
end

# ── Main ──────────────────────────────────────────────────────────────────────
function main()

    println("CUDA.jl FEM Jacobian Test")
    println("Julia version: ", VERSION)
    println("CUDA version:  ", CUDA.versioninfo())
    println()

    # Number of triangular elements (matches Plate_membrane 312x3120 mesh)
    n_elem = 312 * 3120 * 2   # ~1.9M elements
    println("Number of elements: $n_elem")
    println()

    # Generate random triangle node coordinates (6 values per element: x1,y1,x2,y2,x3,y3)
    coords_cpu = rand(Float64, 6, n_elem) .* 100.0

    # ── CPU run ───────────────────────────────────────────────────────────────
    detJ_cpu = zeros(Float64, n_elem)

    # warm up
    compute_detJ_cpu!(detJ_cpu, coords_cpu)

    t_cpu = @elapsed begin
        compute_detJ_cpu!(detJ_cpu, coords_cpu)
    end
    println("CPU time:  $(round(t_cpu * 1000, digits=3)) ms")

    # ── GPU run ───────────────────────────────────────────────────────────────
    coords_gpu = CuArray(coords_cpu)
    detJ_gpu   = CUDA.zeros(Float64, n_elem)

    threads = 256
    blocks  = cld(n_elem, threads)

    # warm up (triggers JIT compilation of kernel)
    @cuda threads=threads blocks=blocks jacobian_kernel!(detJ_gpu, coords_gpu)
    CUDA.synchronize()

    t_gpu = @elapsed begin
        @cuda threads=threads blocks=blocks jacobian_kernel!(detJ_gpu, coords_gpu)
        CUDA.synchronize()
    end
    println("GPU time:  $(round(t_gpu * 1000, digits=3)) ms")
    println("Speedup:   $(round(t_cpu / t_gpu, digits=2))x")
    println()

    # ── Validation ────────────────────────────────────────────────────────────
    detJ_check = Array(detJ_gpu)
    err = norm(detJ_cpu - detJ_check) / norm(detJ_cpu)
    println("Relative error (CPU vs GPU): $err")
    if err < 1e-12
        println("PASSED — results match")
    else
        println("FAILED — results do not match")
    end

end

main()
