# gpu_plate_membrane_surface_pressure.jl
#
# GPU version of the plate membrane (t=2) uniform pressure problem.
# Mirrors: cpu_reference/Plate_membrane_t2_Pressure/TriShellFiniteElement_Membrane_and_Bending_UniformPressure.jl
# Uses gpu_solve() instead of the Ferrite direct sparse solver.
#
# Problem setup (In-plane):
#   Plate:    100 mm × 1000 mm, thickness t = 2 mm || 100 mm
#   Material: E = 200000 MPa, ν = 0.30
#   Load:     uniform pressure p = 1 N/mm², applied as nodal ux (tributary area)
#   BCs:      uy = 0 at midpoint (y=500, x=50)
#             ux = 0 at corners  (y=0, x=0) and (y=1000, x=0)
#             w  = 0 at all nodes on edges y=0 and y=1000

# Problem setup (Out-of-plane):
#   Plate:    100 mm × 1000 mm, thickness t = 2 mm || 100 mm
#   Material: E = 200000 MPa, ν = 0.30
#   Load:     uniform pressure p = 1.00 N/mm² (100mm) 0.001 N/mm² (2 mm), applied as nodal w (tributary area)
#   BCs:      uy = 0 at midpoint (y=500, x=50)
#             ux = 0 at corners  (y=0, x=0) and (y=1000, x=0)
#             w  = 0 at all nodes on edges y=0 and y=1000
#
# Difference from CPU reference:
#   GPU solver requires 6 DOFs/node (adds drilling rotation θz via ip^3 for :θ).
#   The CPU reference uses ip^2 for :θ (5 DOFs/node, no drilling).
#   The drilling DOF is handled internally by the element stiffness kernel
#   via a penalty term and does not affect ux, uy, w, θx, θy results.
#
# Usage:
#   julia --project=. gpu_standard_example/gpu_plate_membrane_surface_pressure.jl

using CUDA
using Ferrite
using Tensors
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "..", "src", "gpu", "gpu_solve.jl"))

function gpu_trishell_uniform_pressure(t, type, p, nel; io=nothing)

    E  = 200000.0
    ν  = 0.30
    left  = Ferrite.Vec(0.0, 0.0)
    right = Ferrite.Vec(100.0, 1000.0)

    grid = generate_grid(Ferrite.Triangle, nel, left, right)

    # Embed 2D grid in 3D (gpu_solve expects Vec{3} node coordinates)
    nodes_3d = [Ferrite.Node(Ferrite.Vec{3,Float64}((n.x[1], n.x[2], 0.0)))
                for n in grid.nodes]
    grid = Ferrite.Grid(grid.cells, nodes_3d)

    ip = Lagrange{RefTriangle, 1}()

    # GPU element kernel produces 18×18 ke → 6 DOFs/node (3 displacement + 3 rotation)
    dh = DofHandler(grid)
    add!(dh, :u, ip^3)   # ux, uy, w
    add!(dh, :θ, ip^3)   # θx, θy, θz  (θz = drilling DOF, handled by penalty)
    close!(dh)

    n_nodes = length(grid.nodes)
    n_elem  = length(grid.cells)
    n_dof   = Ferrite.ndofs(dh)   # = n_nodes * 6

    # ── dof_map (n_elem × 18): Ferrite field-by-field celldofs per element ─────
    dof_map = Matrix{Int32}(undef, n_elem, 18)
    e_idx = 0
    for cell in CellIterator(dh)
        e_idx += 1
        dof_map[e_idx, :] .= Int32.(celldofs(cell))
    end

    # ── element_coords (n_elem × 3 × 3): XYZ of each node per element ─────────
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

    # ── node_to_dofs: node → [ux, uy, w, θx, θy, θz] ─────────────────────────
    # celldofs layout for ip^3 on a triangle (3 nodes):
    #   cdofs[1:9]   = [ux_1, uy_1, w_1,  ux_2, uy_2, w_2,  ux_3, uy_3, w_3]   ← :u field
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

    # ── Identify node sets (same logic as CPU reference) ─────────────────────
    Lx  = right[1]
    Ly  = right[2]
    tol = 1e-8

    middle_nodes   = Int[]
    left_nodes     = Int[]
    right_nodes    = Int[]
    middle_node    = Int[]
    left_top       = Int[]
    right_top      = Int[]
    interior_nodes = Int[]
    edgeX0         = Int[]
    edgeXL         = Int[]

    for (i, node) in enumerate(grid.nodes)
        x = node.x[1]
        y = node.x[2]

        if abs(y - 500.0) < tol;  push!(middle_nodes, i);  end
        if abs(y) < tol;           push!(left_nodes,  i);   end
        if abs(y - Ly) < tol;      push!(right_nodes, i);   end

        if abs(y - 500.0) < tol && abs(x - 50.0) < tol
            push!(middle_node, i)
        end
        if abs(y) < tol && abs(x) < tol
            push!(left_top, i)
        end
        if abs(y - Ly) < tol && abs(x) < tol
            push!(right_top, i)
        end

        if abs(x) < tol
            push!(edgeX0, i)
        elseif abs(x - Lx) < tol
            push!(edgeXL, i)
        elseif x > tol && x < Lx - tol
            push!(interior_nodes, i)
        end
    end

    # ── Force vector (tributary area, same as CPU reference) ──────────────────
    dx = Lx / nel[1]
    dy = Ly / nel[2]
    q  = p * dx * dy

    F = zeros(n_dof)

    if type == "In-plane"
        for n in interior_nodes
            F[node_to_dofs[n][1]] += q
        end
        for n in edgeX0
            F[node_to_dofs[n][1]] += q / 2
        end
        for n in edgeXL
            F[node_to_dofs[n][1]] += q / 2
        end
    elseif type == "Out-of-plane"
        for n in interior_nodes
            F[node_to_dofs[n][3]] += q
        end
        for n in edgeX0
            F[node_to_dofs[n][3]] += q / 2
        end
        for n in edgeXL
            F[node_to_dofs[n][3]] += q / 2
        end
    end

    # ── Dirichlet BCs → fixed_dofs (same constraints as CPU reference) ────────
    fixed_set = Set{Int}()

    for n in middle_node;  push!(fixed_set, node_to_dofs[n][2]);  end   # uy = 0
    for n in left_top;     push!(fixed_set, node_to_dofs[n][1]);  end   # ux = 0
    for n in right_top;    push!(fixed_set, node_to_dofs[n][1]);  end   # ux = 0
    for n in left_nodes;   push!(fixed_set, node_to_dofs[n][3]);  end   # w  = 0
    for n in right_nodes;  push!(fixed_set, node_to_dofs[n][3]);  end   # w  = 0

    # Pin all drilling DOFs (θz, index 6) to zero.
    # θz has no physical meaning for a flat plate; its only stiffness is a tiny
    # penalty term that makes K severely ill-conditioned for iterative solvers.
    for n in 1:n_nodes
        push!(fixed_set, node_to_dofs[n][6])
    end

    fixed_dofs = sort(collect(Int32.(fixed_set)))

    # ── Build mesh_params NamedTuple ──────────────────────────────────────────
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

    # ── GPU solve ──────────────────────────────────────────────────────────────
    t_wall = @elapsed begin
        u, stats = gpu_solve(mesh_params, material, fixed_dofs, F;
                             solver=:custom_pcg, tol=1e-8, max_iter=100_000)
    end

    # ── Extract displacements at y=500 nodes (same as CPU reference) ──────────
    mid_nodes = sort(middle_nodes, by = n -> grid.nodes[n].x[1])

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
        println("$type t_$t $(nel[1])_$(nel[2]): " *
                "Average : ux = $avg_ux uy = $avg_uy; " *
                "Maximum : ux = $max_ux uy = $max_uy")
    end
    if type == "Out-of-plane"
        println("$type t_$t $(nel[1])_$(nel[2]): " *
                "Average : w = $avg_w θx = $avg_θx, θy = $avg_θy; " *
                "Maximum : w = $max_w θx = $max_θx θy = $max_θy")
    end

    println("  solver=$(stats.solver)  iters=$(stats.iters)  " *
            "rel_res=$(round(stats.rel_residual, sigdigits=3))  " *
            "wall=$(round(t_wall*1e3, digits=1)) ms")
    println("  timings(ms): preproc=$(round(tm[:cpu_preproc],digits=1))  upload=$(round(tm[:upload],digits=1))" *
            "  ke=$(round(tm[:ke_kernel],digits=1))  asm=$(round(tm[:assembly],digits=1))" *
            "  bc=$(round(tm[:dirichlet],digits=1))  precond=$(round(tm[:precond],digits=1))" *
            "  solve=$(round(tm[:solve],digits=1))  download=$(round(tm[:download],digits=1))")

    if io !== nothing
        println(io, join([
            type, t, nel[1], nel[2],
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
q       = 0.001   # N/mm — line load intensity

# THICK PLATE:
t_thick = 100.0
q       = 1.0   # N/mm — line load intensity


# IN-PLANE

# THIN PLATE:
t_thin  = 2.0
q       = 1.0   # N/mm — line load intensity

# THICK PLATE:
t_thick = 100.0
q       = 1.0   # N/mm — line load intensity

# num_elem_transverse = [312]
# num_elem_long       = [3120]

# num_elem_transverse = [40]
# num_elem_long       = [400]


# # CHOOSE TEST CASE:
# t_plate = t_thick  

# results_file = joinpath(@__DIR__, "results_surface_pressure.txt")

# # Warm up (triggers JIT compilation, excluded from timing)
# println("Warming up with (4, 40) mesh...")
# gpu_trishell_uniform_pressure(t_plate, "In-plane", p, (4, 40))
# println()

# open(results_file, "w") do io
#     # Header — tab-separated for easy import into Python/Excel/etc.
#     println(io, "# gpu_plate_membrane_surface_pressure results  p=$(p) N/mm²")
#     println(io, join([
#         "type", "t", "nel_x", "nel_y",
#         "avg_ux", "avg_uy", "avg_w", "avg_θx", "avg_θy",
#         "max_ux", "max_uy", "max_w", "max_θx", "max_θy",
#         "solver", "iters", "rel_res",
#         "wall_ms",
#         "preproc_ms", "upload_ms", "ke_ms", "asm_ms",
#         "bc_ms", "precond_ms", "solve_ms", "download_ms",
#     ], "\t"))

#     for i in eachindex(num_elem_transverse)
#         nel = (num_elem_transverse[i], num_elem_long[i])
#         gpu_trishell_uniform_pressure(t_plate, "In-plane",     p, nel; io)
#         gpu_trishell_uniform_pressure(t_plate, "Out-of-plane", p, nel; io)
#     end
# end

# println("\nResults written to: $results_file")





############### CODE FOR DIFFERENT MESH SIZE RESULT COMPAIRSON ###############

# Mesh divisions
num_elem_transverse = [2, 2, 2, 10, 30, 98]
num_elem_long       = [2, 8, 32, 92, 322, 980]

# CHOOSE TEST CASE:
t_plate = t_thick
p = q

results_file = joinpath(@__DIR__, "results_surface_pressure.txt")

# Warm up (triggers JIT compilation, excluded from timing)
println("Warming up with (4, 40) mesh...")
gpu_trishell_uniform_pressure(t_plate, "In-plane", p, (4, 40))
println()

open(results_file, "a") do io

    # Header
    println(io, "# gpu_plate_membrane_surface_pressure results  p=$(p) N/mm²")

    println(io, join([
        "type", "t", "nel_x", "nel_y",
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
        gpu_trishell_uniform_pressure(
            t_plate,
            "In-plane",
            1.0,
            nel,
            io=io
        )

        # Out-of-plane
        gpu_trishell_uniform_pressure(
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
