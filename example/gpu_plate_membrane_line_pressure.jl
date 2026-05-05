# gpu_plate_membrane_line_pressure.jl
#
# GPU version of the plate membrane (t=100) line-load problem.
# Mirrors: cpu_reference/Plate_membrane_t100/TriShellFiniteElement_Membrane_and_Bending.jl
# Uses gpu_solve() instead of the Ferrite direct sparse solver (K \ F).
#
# Problem setup (In-plane):
#   Plate:    100 mm × 1000 mm, thickness t = 100 mm
#   Material: E = 200000 MPa, ν = 0.30
#   Load:     line load q = 1000 N/mm applied in ux at y = 500 (mid-span nodes)
#             distributed by tributary length in x
#   BCs:      uy = 0 at midpoint node (y=500, x=50)
#             ux = 0 at corner nodes  (y=0, x=0) and (y=1000, x=0)
#             w  = 0 at all nodes on edges y=0 and y=1000
#
# Problem setup (Out-of-plane):
#   Same plate and material as above
#   Load:     line load q = 1000 N/mm applied in w (lateral) at y = 500 (mid-span nodes)
#             distributed by tributary length in x
#   BCs:      same as In-plane (w=0 on edges acts as simple support in bending)
#
# Differences from CPU reference:
#   • GPU kernel produces 18×18 ke → 6 DOFs/node (adds drilling θz via ip^3 for :θ).
#     CPU reference uses ip^2 (5 DOFs/node, no drilling).
#     Drilling DOFs are pinned to zero; they do not affect ux/uy/w/θx/θy results.
#   • Force distribution matches CPU reference tributary-length formula exactly.
#
# Usage:
#   julia --project=. gpu_standard_example/gpu_plate_membrane_line_pressure.jl

using CUDA
using Ferrite
using Tensors
using LinearAlgebra
using Statistics
using Dates

include(joinpath(@__DIR__, "..", "src", "gpu", "gpu_solve.jl"))

function write_gpu_run_header(io, t_plate, q)
    slurm_job_id   = get(ENV, "SLURM_JOB_ID", "<not set>")
    slurm_job_name = get(ENV, "SLURM_JOB_NAME", "<not set>")
    slurm_job_gpus = get(ENV, "SLURM_JOB_GPUS", "<not set>")
    cuda_visible   = get(ENV, "CUDA_VISIBLE_DEVICES", "<not set>")

    println(io)
    println(io, "# ============================================================")
    println(io, "# gpu_plate_membrane_line_pressure run")
    println(io, "# started_at=$(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
    println(io, "# host=$(gethostname())")
    println(io, "# slurm_job_id=$(slurm_job_id)")
    println(io, "# slurm_job_name=$(slurm_job_name)")
    println(io, "# slurm_job_gpus=$(slurm_job_gpus)")
    println(io, "# cuda_visible_devices=$(cuda_visible)")
    println(io, "# t=$(t_plate)  q=$(q) N/mm")

    try
        dev = CUDA.device()
        println(io, "# cuda_device=$(dev)")
        println(io, "# cuda_device_name=$(CUDA.name(dev))")
        println(io, "# cuda_capability=$(CUDA.capability(dev))")
        println(io, "# cuda_total_memory=$(CUDA.totalmem(dev)) bytes")
    catch err
        println(io, "# cuda_info_error=$(err)")
    end

    try
        smi = read(`nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader`, String)
        for line in split(chomp(smi), '\n')
            isempty(line) || println(io, "# nvidia_smi_gpu=$line")
        end
    catch err
        println(io, "# nvidia_smi_error=$(err)")
    end

    println(io, "# ============================================================")
end

function gpu_trishell_membrane_t100(t, type, q, nel; io=nothing)

    E   = 200000.0
    ν   = 0.30

    left  = Ferrite.Vec(0.0, 0.0)
    right = Ferrite.Vec(100.0, 1000.0)
    Lx    = right[1]
    Ly    = right[2]
    tol   = 1e-8

    grid = generate_grid(Ferrite.Triangle, nel, left, right)

    # Embed 2D grid in 3D (gpu_solve expects Vec{3} node coordinates)
    nodes_3d = [Ferrite.Node(Ferrite.Vec{3,Float64}((n.x[1], n.x[2], 0.0)))
                for n in grid.nodes]
    grid = Ferrite.Grid(grid.cells, nodes_3d)

    ip = Lagrange{RefTriangle, 1}()

    # 6 DOFs/node: (ux, uy, w) + (θx, θy, θz)
    # θz = drilling DOF, handled by penalty; pinned to zero below.
    dh = DofHandler(grid)
    add!(dh, :u, ip^3)   # ux, uy, w
    add!(dh, :θ, ip^3)   # θx, θy, θz
    close!(dh)

    n_nodes = length(grid.nodes)
    n_elem  = length(grid.cells)
    n_dof   = Ferrite.ndofs(dh)   # = n_nodes * 6

    # ── dof_map (n_elem × 18): Ferrite field-by-field celldofs per element ──────
    dof_map = Matrix{Int32}(undef, n_elem, 18)
    e_idx = 0
    for cell in CellIterator(dh)
        e_idx += 1
        dof_map[e_idx, :] .= Int32.(celldofs(cell))
    end

    # ── element_coords (n_elem × 3 × 3): XYZ of each node per element ──────────
    element_coords = Array{Float64,3}(undef, n_elem, 3, 3)
    e_idx = 0
    for cell in CellIterator(dh)
        e_idx += 1
        for (i, node) in enumerate(cell.nodes)
            element_coords[e_idx, i, 1] = grid.nodes[node].x[1]
            element_coords[e_idx, i, 2] = grid.nodes[node].x[2]
            element_coords[e_idx, i, 3] = grid.nodes[node].x[3]
        end
    end

    # ── node_to_dofs: node → [ux, uy, w, θx, θy, θz] ───────────────────────────
    # celldofs layout for ip^3 on a triangle (3 nodes):
    #   cdofs[1:9]   = [ux_1, uy_1, w_1,  ux_2, uy_2, w_2,  ux_3, uy_3, w_3]  ← :u field
    #   cdofs[10:18] = [θx_1, θy_1, θz_1, θx_2, θy_2, θz_2, θx_3, θy_3, θz_3] ← :θ field
    node_to_dofs = Dict{Int,Vector{Int}}()
    for cell in CellIterator(dh)
        cdofs  = celldofs(cell)
        cnodes = cell.nodes
        for (i, node) in enumerate(cnodes)
            node_to_dofs[node] = [
                cdofs[3*(i-1)+1], cdofs[3*(i-1)+2], cdofs[3*(i-1)+3],
                cdofs[9 + 3*(i-1)+1], cdofs[9 + 3*(i-1)+2], cdofs[9 + 3*(i-1)+3]
            ]
        end
    end

    # ── Identify node sets (same logic as CPU reference) ────────────────────────
    middle_nodes = Int[]
    left_nodes   = Int[]
    right_nodes  = Int[]
    middle_node  = Int[]
    left_top     = Int[]
    right_top    = Int[]

    for (i, node) in enumerate(grid.nodes)
        x = node.x[1]
        y = node.x[2]

        if abs(y - 500.0) < tol;  push!(middle_nodes, i);  end
        if abs(y)         < tol;  push!(left_nodes,   i);  end
        if abs(y - Ly)    < tol;  push!(right_nodes,  i);  end

        if abs(y - 500.0) < tol && abs(x - 50.0) < tol
            push!(middle_node, i)
        end
        if abs(y) < tol && abs(x) < tol
            push!(left_top, i)
        end
        if abs(y - Ly) < tol && abs(x) < tol
            push!(right_top, i)
        end
    end

    # ── Force vector: line load q [N/mm] at y=500, distributed by tributary length
    # Sort mid-span nodes by x so tributary lengths are computed correctly.
    # Matches CPU reference force assembly (tributary-length formula).
    F = zeros(n_dof)
    mid_nodes = sort(middle_nodes, by = n -> grid.nodes[n].x[1])

    for i in eachindex(mid_nodes)
        n  = mid_nodes[i]
        x  = grid.nodes[n].x[1]

        if i == 1
            x_next = grid.nodes[mid_nodes[i+1]].x[1]
            Fx = q * (x_next - x) / 2
        elseif i == length(mid_nodes)
            x_prev = grid.nodes[mid_nodes[i-1]].x[1]
            Fx = q * (x - x_prev) / 2
        else
            x_prev = grid.nodes[mid_nodes[i-1]].x[1]
            x_next = grid.nodes[mid_nodes[i+1]].x[1]
            Fx = q * (x_next - x_prev) / 2
        end

        if type == "In-plane"
            F[node_to_dofs[n][1]] += Fx   # ux DOF
        elseif type == "Out-of-plane"
            F[node_to_dofs[n][3]] += Fx   # w DOF
        end
    end

    # ── Dirichlet BCs → fixed_dofs ───────────────────────────────────────────────
    fixed_set = Set{Int}()

    for n in middle_node;  push!(fixed_set, node_to_dofs[n][2]);  end   # uy = 0
    for n in left_top;     push!(fixed_set, node_to_dofs[n][1]);  end   # ux = 0
    for n in right_top;    push!(fixed_set, node_to_dofs[n][1]);  end   # ux = 0
    for n in left_nodes;   push!(fixed_set, node_to_dofs[n][3]);  end   # w  = 0
    for n in right_nodes;  push!(fixed_set, node_to_dofs[n][3]);  end   # w  = 0

    # Pin all drilling DOFs (θz, index 6) — no physical stiffness, only penalty.
    for n in 1:n_nodes
        push!(fixed_set, node_to_dofs[n][6])
    end

    fixed_dofs = sort(collect(Int32.(fixed_set)))

    # ── Build mesh_params NamedTuple ─────────────────────────────────────────────
    node_coords = Matrix{Float64}(undef, n_nodes, 3)
    for (i, node) in enumerate(grid.nodes)
        node_coords[i, 1] = node.x[1]
        node_coords[i, 2] = node.x[2]
        node_coords[i, 3] = node.x[3]
    end

    connectivity = Matrix{Int32}(undef, n_elem, 3)
    for (e, cell) in enumerate(grid.cells)
        for (j, node) in enumerate(cell.nodes)
            connectivity[e, j] = Int32(node)
        end
    end

    mesh_params = (
        n_elem         = n_elem,
        n_nodes        = n_nodes,
        n_dof          = n_dof,
        node_coords    = node_coords,
        connectivity   = connectivity,
        dof_map        = dof_map,
        element_coords = element_coords,
    )

    material = (E=E, nu=ν, t=t)

    # ── GPU solve ────────────────────────────────────────────────────────────────
    t_wall = @elapsed begin
        u, stats = gpu_solve(mesh_params, material, fixed_dofs, F;
                             solver=:custom_pcg, tol=1e-8, max_iter=100_000)
    end

    # ── Extract displacements at y=500 nodes (matches CPU reference output) ──────
    ux_vals = Float64[]
    uy_vals = Float64[]
    w_vals  = Float64[]
    θx_vals = Float64[]
    θy_vals = Float64[]

    for n in mid_nodes
        d = node_to_dofs[n]
        push!(ux_vals, u[d[1]])
        push!(uy_vals, u[d[2]])
        push!(w_vals,  u[d[3]])
        push!(θx_vals, u[d[4]])
        push!(θy_vals, u[d[5]])
    end

    tm = stats.timings
    avg_ux = mean(ux_vals);  max_ux = maximum(abs, ux_vals)
    avg_uy = mean(uy_vals);  max_uy = maximum(abs, uy_vals)
    avg_w  = mean(w_vals);   max_w  = maximum(abs, w_vals)
    avg_θx = mean(θx_vals);  max_θx = maximum(abs, θx_vals)
    avg_θy = mean(θy_vals);  max_θy = maximum(abs, θy_vals)

    if type == "In-plane"
        println("$type t_100 $(nel[1])_$(nel[2]): " *
                "Average : ux = $avg_ux uy = $avg_uy; " *
                "Maximum : ux = $max_ux uy = $max_uy")
    elseif type == "Out-of-plane"
        println("$type t_100 $(nel[1])_$(nel[2]): " *
                "Average : w = $avg_w θx = $avg_θx, θy = $avg_θy; " *
                "Maximum : w = $max_w θx = $max_θx θy = $max_θy")
    end

    println("  solver=$(stats.solver)  iters=$(stats.iters)  " *
            "rel_res=$(round(stats.rel_residual, sigdigits=3))  " *
            "wall=$(round(t_wall*1e3, digits=1)) ms")
    println("  timings(ms): preproc=$(round(tm[:cpu_preproc],digits=1))" *
            "  upload=$(round(tm[:upload],digits=1))" *
            "  ke=$(round(tm[:ke_kernel],digits=1))" *
            "  asm=$(round(tm[:assembly],digits=1))" *
            "  bc=$(round(tm[:dirichlet],digits=1))" *
            "  precond=$(round(tm[:precond],digits=1))" *
            "  solve=$(round(tm[:solve],digits=1))" *
            "  download=$(round(tm[:download],digits=1))")

    if io !== nothing
        println(io, join([
            type, nel[1], nel[2],
            avg_ux, avg_uy, avg_w, avg_θx, avg_θy,
            max_ux, max_uy, max_w, max_θx, max_θy,
            stats.solver, stats.iters,
            round(stats.rel_residual, sigdigits=6),
            round(t_wall*1e3, digits=3),
            round(tm[:cpu_preproc], digits=3), round(tm[:upload],    digits=3),
            round(tm[:ke_kernel],   digits=3), round(tm[:assembly],  digits=3),
            round(tm[:dirichlet],   digits=3), round(tm[:precond],   digits=3),
            round(tm[:solve],       digits=3), round(tm[:download],  digits=3),
        ], "\t"))
        flush(io)
    end

    return u, stats
end

# ── Main ──────────────────────────────────────────────────────────────────────

# TEST CASES

# OUT-OF-PLANE

# THIN PLATE:
t_thin  = 2.0
q       = 1.0   # N/mm — line load intensity

# THICK PLATE:
t_thick = 100.0
q       = 1000.0   # N/mm — line load intensity


# IN-PLANE

# THIN PLATE:
t_thin  = 2.0
q       = 1000.0   # N/mm — line load intensity

# THICK PLATE:
t_thick = 100.0
q       = 1000.0   # N/mm — line load intensity

# # num_elem_transverse = [312]
# # num_elem_long       = [3120]

# # num_elem_transverse = [2, 2, 2, 10, 30, 98, 312]
# # num_elem_long       = [2, 8, 32, 92, 322, 980, 3120]



# # num_elem_transverse = [40]
# # num_elem_long       = [400]


# # CHOOSE TEST CASE:
# t_plate = t_thin   

# results_file = joinpath(@__DIR__, "results_line_pressure.txt")

# # Warm up (triggers JIT compilation, excluded from timing)
# println("Warming up with (4, 40) mesh...")
# gpu_trishell_membrane_t100(t_plate, "In-plane", q, (4, 40))
# println()

# open(results_file, "w") do io
#     # Header — tab-separated for easy import into Python/Excel/etc.
#     println(io, "# gpu_plate_membrane_line_pressure results  t=$(t_plate)  q=$(q) N/mm")
#     println(io, join([
#         "type", "nel_x", "nel_y",
#         "avg_ux", "avg_uy", "avg_w", "avg_θx", "avg_θy",
#         "max_ux", "max_uy", "max_w", "max_θx", "max_θy",
#         "solver", "iters", "rel_res",
#         "wall_ms",
#         "preproc_ms", "upload_ms", "ke_ms", "asm_ms",
#         "bc_ms", "precond_ms", "solve_ms", "download_ms",
#     ], "\t"))

#     for i in eachindex(num_elem_transverse)
#         nel = (num_elem_transverse[i], num_elem_long[i])
#         gpu_trishell_membrane_t100(t_plate, "In-plane",     1000.0, nel; io)
#         gpu_trishell_membrane_t100(t_plate, "Out-of-plane", 1.0, nel; io)
#     end
# end

# println("\nResults written to: $results_file")



############### CODE FOR DIFFERENT MESH SIZE RESULT COMPAIRSON ###############

# Mesh divisions
num_elem_transverse = [2, 2, 2, 10, 30, 98]
num_elem_long       = [2, 8, 32, 92, 322, 980]

# CHOOSE TEST CASE:
t_plate = t_thin

results_file = joinpath(@__DIR__, "results_line_pressure_mesh_thin.txt")

# Warm up (triggers JIT compilation, excluded from timing)
println("Warming up with (4, 40) mesh...")
gpu_trishell_membrane_t100(t_plate, "In-plane", 1000.0, (4, 40))
println()

open(results_file, "a") do io

    write_gpu_run_header(io, t_plate, q)
    flush(io)

    println(io, join([
        "type", "nel_x", "nel_y",
        "avg_ux", "avg_uy", "avg_w", "avg_θx", "avg_θy",
        "max_ux", "max_uy", "max_w", "max_θx", "max_θy",
        "solver", "iters", "rel_res",
        "wall_ms",
        "preproc_ms", "upload_ms", "ke_ms", "asm_ms",
        "bc_ms", "precond_ms", "solve_ms", "download_ms"
    ], "\t"))
    flush(io)

    # Loop through all mesh sizes
    for (nel_x, nel_y) in zip(num_elem_transverse, num_elem_long)

        nel = (nel_x, nel_y)

        println("Running mesh: $(nel_x) x $(nel_y)")

        # In-plane
        gpu_trishell_membrane_t100(
            t_plate,
            "In-plane",
            1000.0,
            nel,
            io=io
        )

        # Out-of-plane
        gpu_trishell_membrane_t100(
            t_plate,
            "Out-of-plane",
            1.0,
            nel,
            io=io
        )

        println()
    end
end

println("\nResults written to: $results_file")
