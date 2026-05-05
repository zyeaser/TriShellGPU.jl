using CUDA
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# INFRASTRUCTURE BENCHMARK — CUDA.jl kernel launch overhead + memory bandwidth
#
# PURPOSE: Measure raw GPU vs CPU throughput for a trivial per-element kernel
#          (2D Jacobian determinant) to establish a baseline before any FEM
#          physics is implemented. Also verifies CSV output and Slurm integration.
#          This is NOT a benchmark of the actual 3D shell element stiffness kernel.
#
# WHAT IT MEASURES: Throughput of a minimal kernel (6 reads + 3 arithmetic ops
#                   per element). Dominated by memory bandwidth, not compute.
#                   The actual element_stiffness_kernel! is ~100x more work per
#                   element and will show different roofline behavior.
#
# WHAT IT DOES NOT MEASURE:
#   - element_stiffness_kernel! throughput (18x18 ke, IP3+IP6, rotation, condensation)
#   - assembly_kernel! throughput (atomic adds, scatter_idx)
#   - pcg_solver throughput (SpMV, dot, axpy)
#   For those, see: benchmarks/run_bending.jl, run_membrane.jl
#
# ─────────────────────────────────────────────────────────────────────────────
function jacobian_kernel!(detJ_out, coords)
    e = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    e > size(coords, 2) && return

    x1 = coords[1, e];  y1 = coords[2, e]
    x2 = coords[3, e];  y2 = coords[4, e]
    x3 = coords[5, e];  y3 = coords[6, e]

    J11 = x2 - x1;  J12 = y2 - y1
    J21 = x3 - x1;  J22 = y3 - y1

    detJ_out[e] = J11 * J22 - J12 * J21
    return
end

# ─────────────────────────────────────────────────────────────────────────────
# CPU reference
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
function main()

    gpu_label = length(ARGS) >= 1 ? ARGS[1] : "unknown"
    output_csv = length(ARGS) >= 2 ? ARGS[2] : "results.csv"

    n_elem = 312 * 3120 * 2   # 1,946,880 elements (matches full mesh)

    println("=" ^ 60)
    println("GPU Benchmark — FEM Jacobian Kernel")
    println("GPU label    : $gpu_label")
    println("Julia version: $(VERSION)")
    println("Elements     : $n_elem")
    println("=" ^ 60)

    # GPU info
    dev = CUDA.device()
    gpu_name   = CUDA.name(dev)
    gpu_mem_gb = round(CUDA.totalmem(dev) / 1024^3, digits=2)
    println("Device: $gpu_name  ($gpu_mem_gb GiB)")
    println()

    # Generate random element coordinates
    coords_cpu = rand(Float64, 6, n_elem) .* 100.0

    # ── CPU benchmark ─────────────────────────────────────────────────────────
    detJ_cpu = zeros(Float64, n_elem)
    compute_detJ_cpu!(detJ_cpu, coords_cpu)   # warm up

    n_runs = 10
    t_cpu_total = 0.0
    for _ in 1:n_runs
        t_cpu_total += @elapsed compute_detJ_cpu!(detJ_cpu, coords_cpu)
    end
    t_cpu_ms = (t_cpu_total / n_runs) * 1000

    println("CPU time (avg $n_runs runs): $(round(t_cpu_ms, digits=3)) ms")

    # ── GPU benchmark ─────────────────────────────────────────────────────────
    coords_gpu = CuArray(coords_cpu)
    detJ_gpu   = CUDA.zeros(Float64, n_elem)

    threads = 256
    blocks  = cld(n_elem, threads)

    # warm up
    @cuda threads=threads blocks=blocks jacobian_kernel!(detJ_gpu, coords_gpu)
    CUDA.synchronize()

    t_gpu_total = 0.0
    for _ in 1:n_runs
        t_gpu_total += @elapsed begin
            @cuda threads=threads blocks=blocks jacobian_kernel!(detJ_gpu, coords_gpu)
            CUDA.synchronize()
        end
    end
    t_gpu_ms = (t_gpu_total / n_runs) * 1000

    println("GPU time (avg $n_runs runs): $(round(t_gpu_ms, digits=3)) ms")

    speedup = t_cpu_ms / t_gpu_ms
    println("Speedup: $(round(speedup, digits=2))x")

    # ── Validation ────────────────────────────────────────────────────────────
    detJ_check = Array(detJ_gpu)
    err = norm(detJ_cpu - detJ_check) / norm(detJ_cpu)
    passed = err < 1e-12
    println("Relative error: $err  →  $(passed ? "PASSED" : "FAILED")")
    println()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    write_header = !isfile(output_csv)
    open(output_csv, "a") do f
        if write_header
            println(f, "gpu_label,gpu_name,gpu_mem_gb,n_elements,cpu_time_ms,gpu_time_ms,speedup,error,passed")
        end
        println(f, "$gpu_label,$gpu_name,$gpu_mem_gb,$n_elem,$(round(t_cpu_ms,digits=3)),$(round(t_gpu_ms,digits=3)),$(round(speedup,digits=2)),$err,$passed")
    end
    println("Results appended to: $output_csv")
end

main()
