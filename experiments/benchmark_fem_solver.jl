# benchmark_fem_solver.jl
#
# Benchmarks the GPU element stiffness kernel (element_stiffness_kernel!)
# across a range of mesh sizes with THREE baselines:
#
#   1. GPU   — element_stiffness_kernel! on CUDA device
#   2. CPU   — exact same algorithm, serial loop, using device_functions.jl helpers
#   3. Ferrite CPU — TriShellFiniteElement.assemble_global_Ke! (the real reference)
#
# NOTE on Ferrite timing:
#   local_elastic_stiffness_matrix! contains several println() debug statements
#   (x, Dm, ke_m, ke_b) that fire every element.  Stdout is redirected to devnull
#   during timing to avoid flooding the log, but the string-formatting overhead
#   is still included in the measured time — so Ferrite timings are PESSIMISTIC
#   (slower than they would be with the printlns removed).
#   Because of this, Ferrite is only benchmarked up to 10K elements.
#
# Called by run_fem_solver.sh as:
#   julia --project=. experiments/benchmark_fem_solver.jl <GPU_LABEL> <RESULTS_CSV>
#
# CSV columns:
#   gpu_label, gpu_name, gpu_mem_gb, cpu_model,
#   n_elements,
#   gpu_time_ms,
#   serial_cpu_time_ms, serial_speedup,
#   ferrite_cpu_time_ms, ferrite_speedup,
#   threads_per_block, julia_version

using CUDA
using StaticArrays
using LinearAlgebra
using Printf
using Ferrite
using Tensors
using TriShellFiniteElement

# ── Load GPU kernels ───────────────────────────────────────────────────────────
# All @inline helpers in device_functions.jl work on CPU too (no CUDA intrinsics).
include(joinpath(@__DIR__, "..", "src", "gpu", "device_functions.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu", "element_stiffness_kernel.jl"))

# =============================================================================
# BASELINE 1 — Serial CPU loop (same algorithm as GPU, single thread)
# =============================================================================

function element_ke_cpu!(ke_all::Array{Float64,3}, e::Int,
                          coords::Array{Float64,3},
                          E::Float64, ν::Float64, t::Float64)
    P1x = coords[e,1,1]; P1y = coords[e,1,2]; P1z = coords[e,1,3]
    P2x = coords[e,2,1]; P2y = coords[e,2,2]; P2z = coords[e,2,3]
    P3x = coords[e,3,1]; P3y = coords[e,3,2]; P3z = coords[e,3,3]

    j1x, j1y, j1z,
    j2x, j2y, j2z,
    j3x, j3y, j3z = local_to_global_rotation(P1x, P1y, P1z,
                                               P2x, P2y, P2z,
                                               P3x, P3y, P3z)

    lx1, ly1, lx2, ly2, lx3, ly3 =
        project_to_local(P1x, P1y, P1z, P2x, P2y, P2z, P3x, P3y, P3z,
                         j1x, j1y, j1z, j2x, j2y, j2z)

    Jinv11, Jinv12, Jinv21, Jinv22, detJ =
        compute_jacobian(lx1, ly1, lx2, ly2, lx3, ly3)

    Dm  = constitutive_membrane(E, ν, t)
    Db  = constitutive_bending(E, ν, t)
    gst = constitutive_shear_diag(E, ν, t)

    km = MArray{Tuple{6,6},   Float64, 2,  36}(undef); fill!(km,  0.0)
    kb = MArray{Tuple{9,9},   Float64, 2,  81}(undef); fill!(kb,  0.0)
    ks = MArray{Tuple{12,12}, Float64, 2, 144}(undef); fill!(ks,  0.0)

    let
        _, dN_dξ₁_3, dN_dξ₂_3 = shape_functions_ip3(QR1_ξ₁[1], QR1_ξ₂[1])
        dN_dx3, dN_dy3 = phys_derivs(dN_dξ₁_3, dN_dξ₂_3, Jinv11, Jinv12, Jinv21, Jinv22)
        wdetJ1 = QR1_w[1] * detJ
        btdb_membrane!(km, B_membrane(dN_dx3, dN_dy3), Dm, wdetJ1)
        btdb_bending!( kb, B_bending( dN_dx3, dN_dy3), Db, wdetJ1)
    end

    for q in 1:4
        ξ₁q = QR3_ξ₁[q]; ξ₂q = QR3_ξ₂[q]
        N_ip3, _, _             = shape_functions_ip3(ξ₁q, ξ₂q)
        _, dN_dξ₁_6, dN_dξ₂_6 = shape_functions_ip6(ξ₁q, ξ₂q)
        dN_dx6, dN_dy6 = phys_derivs(dN_dξ₁_6, dN_dξ₂_6, Jinv11, Jinv12, Jinv21, Jinv22)
        wdetJq = QR3_w[q] * detJ
        btdb_shear!(ks, B_shear(N_ip3, dN_dx6, dN_dy6), gst, wdetJq)
    end

    ks_ii     = MArray{Tuple{3,3},Float64,2,9}(undef)
    ks_ii_inv = MArray{Tuple{3,3},Float64,2,9}(undef)
    ks_cond   = MArray{Tuple{9,9},Float64,2,81}(undef)
    @inbounds for i in 1:3, j in 1:3; ks_ii[i,j] = ks[9+i, 9+j]; end
    inv3x3!(ks_ii_inv, ks_ii)
    @inbounds for r in 1:9, c in 1:9
        s = ks[r,c]
        for k in 1:3, l in 1:3; s -= ks[r,9+k] * ks_ii_inv[k,l] * ks[9+l,c]; end
        ks_cond[r,c] = s
    end

    ke_bs = MArray{Tuple{9,9},Float64,2,81}(undef)
    @inbounds for i in 1:9, j in 1:9; ke_bs[i,j] = ks_cond[i,j] + kb[i,j]; end

    ke_local = MArray{Tuple{18,18},Float64,2,324}(undef); fill!(ke_local, 0.0)
    @inbounds for r in 1:6, c in 1:6
        ke_local[INDUV18[r], INDUV18[c]] = km[r,c]
    end
    @inbounds for r in 1:9, c in 1:9
        ke_local[INDWT18[r], INDWT18[c]] = ke_bs[r,c]
    end
    @inbounds stif = min(ke_bs[2,2], min(ke_bs[3,3], min(ke_bs[5,5],
                     min(ke_bs[6,6], min(ke_bs[8,8], ke_bs[9,9]))))) / 100.0
    ke_local[6,6] = stif; ke_local[12,12] = stif; ke_local[18,18] = stif

    ke_global = MArray{Tuple{18,18},Float64,2,324}(undef)
    rotate_ke18!(ke_global, ke_local, j1x, j1y, j1z, j2x, j2y, j2z, j3x, j3y, j3z)

    @inbounds for i in 1:18, j in 1:18
        ke_all[e, i, j] = ke_global[IND_FIELD[i], IND_FIELD[j]]
    end
    return nothing
end

function ke_serial_loop!(ke_all, coords, E, ν, t)
    for e in 1:size(coords, 1)
        element_ke_cpu!(ke_all, e, coords, E, ν, t)
    end
end

# =============================================================================
# BASELINE 2 — Ferrite + TriShellFiniteElement CPU reference
#
# Builds a proper Ferrite 3-D triangular mesh and calls assemble_global_Ke!.
# Uses QuadratureRule{RefTriangle}(3) to match the GPU's 4-point degree-3 rule.
# Stdout is redirected to devnull to suppress debug println inside
# local_elastic_stiffness_matrix! — but string-formatting overhead remains.
# =============================================================================

function build_ferrite_grid(n_elem_target::Int)
    # generate_grid(Triangle, (nx,ny)) → 2*nx*ny triangles in unit square
    nx = ceil(Int, sqrt(n_elem_target / 2))
    ny = ceil(Int, n_elem_target / (2 * nx))

    # Start with a 2-D grid then lift to 3-D (z = 0) — required by
    # TriShellFiniteElement which expects Vec{3} node coordinates.
    grid2d = generate_grid(Triangle, (nx, ny),
                           Vec((0.0, 0.0)), Vec(Float64(nx), Float64(ny)))
    nodes3d = [Node(Tensors.Vec{3,Float64}((n.x[1], n.x[2], 0.0)))
               for n in grid2d.nodes]
    grid3d = Grid(grid2d.cells, nodes3d)
    return grid3d, 2 * nx * ny   # grid, actual element count
end

function ferrite_assemble_once!(K, dh, qr1, qr3, ip3, ip6, E, ν, t)
    # Redirect stdout to suppress per-element println debug output
    redirect_stdout(devnull) do
        TriShellFiniteElement.assemble_global_Ke!(K, dh, qr1, qr3, ip3, ip6, E, ν, t)
    end
end

function run_ferrite_benchmark(n_elem_target, E, ν, t; n_runs=1)
    grid, n_actual = build_ferrite_grid(n_elem_target)

    ip  = Lagrange{RefTriangle, 1}()
    ip3 = TriShellFiniteElement.IP3()
    ip6 = TriShellFiniteElement.IP6()
    qr1 = QuadratureRule{RefTriangle}(1)   # 1-point centroid rule
    qr3 = QuadratureRule{RefTriangle}(3)   # degree-3, 4-point — matches GPU

    dh = DofHandler(grid)
    add!(dh, :u, ip^3)   # u, v, w  (3 displacement DOFs per node)
    add!(dh, :θ, ip^3)   # θx, θy, θz  (3 rotation DOFs per node, incl. drilling)
    close!(dh)

    K = allocate_matrix(dh)

    # Warm-up (also triggers JIT compilation)
    ferrite_assemble_once!(K, dh, qr1, qr3, ip3, ip6, E, ν, t)

    t_total = 0.0
    for _ in 1:n_runs
        fill!(K.nzval, 0.0)   # reset sparse values only — no re-allocation
        t_total += @elapsed ferrite_assemble_once!(K, dh, qr1, qr3, ip3, ip6, E, ν, t)
    end

    return (t_total / n_runs) * 1e3, n_actual   # ms, actual n_elem
end

# =============================================================================
# Mesh generator for GPU / serial CPU (flat plate, raw coord array)
# =============================================================================

function build_flat_mesh(n_elem_target::Int)
    nx = ceil(Int, sqrt(n_elem_target / 2))
    ny = ceil(Int, n_elem_target / (2 * nx))
    n_actual = 2 * nx * ny
    coords = Array{Float64,3}(undef, n_actual, 3, 3)
    e = 0
    for iy in 1:ny, ix in 1:nx
        x0, y0 = Float64(ix - 1), Float64(iy - 1)
        e += 1
        coords[e, 1, :] .= [x0,   y0,   0.0]
        coords[e, 2, :] .= [x0+1, y0,   0.0]
        coords[e, 3, :] .= [x0,   y0+1, 0.0]
        e += 1
        coords[e, 1, :] .= [x0+1, y0,   0.0]
        coords[e, 2, :] .= [x0+1, y0+1, 0.0]
        coords[e, 3, :] .= [x0,   y0+1, 0.0]
    end
    return coords
end

# =============================================================================
# Main
# =============================================================================

function main()
    gpu_label  = length(ARGS) >= 1 ? ARGS[1] : "unknown"
    output_csv = length(ARGS) >= 2 ? ARGS[2] :
                    joinpath(@__DIR__, "benchmark_results_fem.csv")

    # GPU / serial CPU / Ferrite: full size sweep
    n_elem_sizes_gpu = [1_000, 10_000, 100_000, 500_000, 1_000_000]

    threads_per_block = 256
    n_runs_gpu = 5
    n_runs_ferrite = 1

    println("=" ^ 60)
    println("FEM Solver Benchmark — GPU vs CPU vs Ferrite reference")
    println("GPU label    : $gpu_label")
    println("Julia version: $(VERSION)")
    println("=" ^ 60)

    dev        = CUDA.device()
    gpu_name   = CUDA.name(dev)
    gpu_mem_gb = round(CUDA.totalmem(dev) / 1024^3, digits=2)
    println("GPU: $gpu_name  ($gpu_mem_gb GiB)")

    cpu_model = try
        strip(read(pipeline(`grep -m1 "model name" /proc/cpuinfo`,
                            `sed 's/.*: //'`), String))
    catch
        Sys.cpu_info()[1].model
    end
    println("CPU: $cpu_model")
    println()

    # ── CSV setup ────────────────────────────────────────────────────────────
    expected_header =
        "gpu_label,gpu_name,gpu_mem_gb,cpu_model,n_elements," *
        "gpu_time_ms,serial_cpu_time_ms,serial_speedup," *
        "ferrite_cpu_time_ms,ferrite_speedup," *
        "threads_per_block,julia_version"

    if isfile(output_csv)
        existing_header = readline(output_csv)
        if existing_header != expected_header
            stale = output_csv * ".bak_" * string(round(Int, time()))
            mv(output_csv, stale)
            @warn "CSV header mismatch — old file moved to $stale, starting fresh"
        end
    end

    csv_io = open(output_csv, "a")
    if !isfile(output_csv) || filesize(output_csv) == 0
        println(csv_io, expected_header)
    end

    E = 2.1e11; ν = 0.3; t = 0.01   # steel, 1 cm plate

    # ── Pre-run Ferrite benchmarks (same sweep as GPU) ────────────────────────
    println("── Ferrite CPU baseline (sizes: $n_elem_sizes_gpu) ─────────────────")
    println("   (stdout suppressed; includes println overhead in reference code)")
    ferrite_times = Dict{Int,Float64}()   # n_elem → ms
    for n_target in n_elem_sizes_gpu
        t_ms, n_actual = run_ferrite_benchmark(n_target, E, ν, t; n_runs=n_runs_ferrite)
        ferrite_times[n_actual] = t_ms
        @printf("  n=%7d : %8.1f ms\n", n_actual, t_ms)
    end
    println()

    # ── GPU + serial CPU sweep ────────────────────────────────────────────────
    println("── GPU + serial CPU sweep ───────────────────────────────────────────")
    for n_target in n_elem_sizes_gpu
        coords_cpu = build_flat_mesh(n_target)
        n_elem     = size(coords_cpu, 1)

        println("── n_elem = $n_elem ─────────────────────────────────────")

        # GPU
        coords_d = CuArray(coords_cpu)
        ke_gpu_d = CUDA.zeros(Float64, n_elem, 18, 18)
        blocks   = cld(n_elem, threads_per_block)

        @cuda threads=threads_per_block blocks=blocks element_stiffness_kernel!(
            ke_gpu_d, coords_d, E, ν, t, Int32(n_elem))
        CUDA.synchronize()

        t_gpu = 0.0
        for _ in 1:n_runs_gpu
            t_gpu += @elapsed begin
                @cuda threads=threads_per_block blocks=blocks element_stiffness_kernel!(
                    ke_gpu_d, coords_d, E, ν, t, Int32(n_elem))
                CUDA.synchronize()
            end
        end
        t_gpu_ms = (t_gpu / n_runs_gpu) * 1e3
        @printf("  GPU:        %8.3f ms  (avg %d runs)\n", t_gpu_ms, n_runs_gpu)

        # Serial CPU
        ke_cpu_out = zeros(Float64, n_elem, 18, 18)
        ke_serial_loop!(ke_cpu_out, coords_cpu, E, ν, t)   # warm-up

        t_serial = 0.0
        for _ in 1:n_runs_gpu
            GC.gc()
            t_serial += @elapsed ke_serial_loop!(ke_cpu_out, coords_cpu, E, ν, t)
        end
        t_serial_ms   = (t_serial / n_runs_gpu) * 1e3
        serial_speedup = t_serial_ms / t_gpu_ms
        @printf("  Serial CPU: %8.3f ms  (avg %d runs)  →  %.1fx speedup\n",
                t_serial_ms, n_runs_gpu, serial_speedup)

        # Ferrite CPU (look up pre-computed value, or mark N/A)
        t_ferrite_ms   = get(ferrite_times, n_elem, NaN)
        ferrite_speedup = isnan(t_ferrite_ms) ? NaN : t_ferrite_ms / t_gpu_ms
        if isnan(t_ferrite_ms)
            println("  Ferrite CPU: N/A (not benchmarked at this size)")
        else
            @printf("  Ferrite CPU: %8.1f ms  (1 run)           →  %.1fx speedup\n",
                    t_ferrite_ms, ferrite_speedup)
        end
        println()

        # CSV
        ferrite_str  = isnan(t_ferrite_ms)   ? "N/A" : string(round(t_ferrite_ms,   digits=1))
        fspeedup_str = isnan(ferrite_speedup) ? "N/A" : string(round(ferrite_speedup, digits=1))
        println(csv_io,
            "$gpu_label,$gpu_name,$gpu_mem_gb,\"$cpu_model\",$n_elem," *
            "$(round(t_gpu_ms,    digits=3))," *
            "$(round(t_serial_ms, digits=3)),$(round(serial_speedup, digits=2))," *
            "$ferrite_str,$fspeedup_str," *
            "$threads_per_block,$(VERSION)")

        CUDA.unsafe_free!(coords_d)
        CUDA.unsafe_free!(ke_gpu_d)
    end

    close(csv_io)
    println("Results appended → $output_csv")
    println("=" ^ 60)
end

main()
