using TriShellFiniteElement
using Ferrite
using Tensors

E = 200000.0
t = 100.0
ν = 0.30

type = "In-plane"       # In-plane or Out-of-plane

q = 1000  # N/mm           # Total load intensity


left  = Ferrite.Vec(0.0, 0.0)
right = Ferrite.Vec(100.0, 1000.0)
nel = (312,3120)
# nel = (4,40)

grid = generate_grid(Ferrite.Triangle, nel, left, right)

##########################
# Embed 2D grid in 3D (required by TriShellFiniteElement which uses cross products)
nodes_3d = [Ferrite.Node(Ferrite.Vec{3,Float64}((n.x[1], n.x[2], 0.0))) for n in grid.nodes]
grid = Ferrite.Grid(grid.cells, nodes_3d)
##########################




ip = Lagrange{RefTriangle,1}()
ip6 = TriShellFiniteElement.IP6()
ip3 = TriShellFiniteElement.IP3()
qr1 = QuadratureRule{RefTriangle}(1)  
qr3 = QuadratureRule{RefTriangle}(2)  


dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

K = allocate_matrix(dh)

# println(K)
# K = TriShellFiniteElement.assemble_global!(K, dh, qr1, qr3, ip3, ip6, E, ν, t)


assembly_start = time_ns()
K = TriShellFiniteElement.assemble_global_Ke!(K, dh, qr1, qr3, ip3, ip6, E, ν, t)
assembly_time = (time_ns() - assembly_start) / 1e9
# println(Matrix(K))


# Extract and define cells and nodes

cell = first(CellIterator(dh))


cells = grid.cells
nodes = grid.nodes
length(nodes)

middle_nodes = Int[]
for i in 1:length(grid.nodes)
    if grid.nodes[i].x[2] == 500.0
        push!(middle_nodes, i)
    end
end

left_nodes = Int[]
for i in 1:length(grid.nodes)
    if grid.nodes[i].x[2] == 0.0
        push!(left_nodes, i)
    end
end

right_nodes = Int[]
for i in 1:length(grid.nodes)
    if grid.nodes[i].x[2] == 1000.0
        push!(right_nodes, i)
    end
end

middle_node  = Int[]
left_top  = Int[]
right_top = Int[]

tol = 1e-8

for (i, node) in enumerate(grid.nodes)
    x, y = node.x

    if     y == 500.0  && x == 50.0 
        push!(middle_node, i)
    elseif y == 0.0   && x == 0.0
        push!(left_top, i)
    elseif y == 1000.0 && x == 0.0
        push!(right_top, i)
    end
end

#### Mapping nodes to dof
node_to_dofs = Dict{Int,Vector{Int}}()
for cell in CellIterator(dh)
    cdofs = Ferrite.celldofs(cell)
    cnodes = cell.nodes

    for (i, node) in enumerate(cnodes)
        # each node has 5 DOFs (ux, uy, w, θx, θy)
        dofs = cdofs[5*i-4 : 5*i]
        node_to_dofs[node] = dofs
    end
end


n_dofs = Ferrite.ndofs(dh)
F = zeros(n_dofs)


mid_nodes = sort(middle_nodes, by = n -> grid.nodes[n].x[2])
for i in eachindex(mid_nodes)
    n = mid_nodes[i]
    x = grid.nodes[n].x[1]

    if i == 1
        x1 = grid.nodes[mid_nodes[i+1]].x[1]
        Fx = q * (x1 - x) / 2
    elseif i == length(mid_nodes)
        x2 = grid.nodes[mid_nodes[i-1]].x[1]
        Fx = q * (x - x2) / 2
    else
        x1 = grid.nodes[mid_nodes[i-1]].x[1]
        x2 = grid.nodes[mid_nodes[i+1]].x[1]
        Fx = q * (x2 - x1) / 2
    end
    if type == "In-plane"
        ux_dof = node_to_dofs[n][1]
    elseif type == "Out-of-plane"
        ux_dof = node_to_dofs[n][3]
    end
        F[ux_dof] += Fx
    # println(F[ux_dof])
end


ch = ConstraintHandler(dh)

# uy restrained at middle node
add!(ch, Dirichlet(:u, middle_node,  (x, t) -> [0.0], [2]))

# ux restrained at top corner nodes
add!(ch, Dirichlet(:u, left_top,  (x, t) -> [0.0], [1]))
add!(ch, Dirichlet(:u, right_top, (x, t) -> [0.0], [1]))
add!(ch, Dirichlet(:u, left_nodes,  (x, t) -> [0.0], [3]))
add!(ch, Dirichlet(:u, right_nodes, (x, t) -> [0.0], [3]))
close!(ch)


apply!(K, F, ch)
u = K \ F
apply!(u, ch)

for n in middle_node
    for cell in CellIterator(dh)
        idx = findfirst(==(n), cell.nodes)
        idx === nothing && continue

        cdofs = celldofs(cell)

        ux = u[cdofs[3*(idx-1) + 1]]
        uy = u[cdofs[3*(idx-1) + 2]]
        w  = u[cdofs[3*(idx-1) + 3]]
        θx = u[cdofs[9 + 2*(idx-1) + 1]]
        θy = u[cdofs[9 + 2*(idx-1) + 2]]

        println(
            "Middle_node ID $n : ux = $ux, uy = $uy, w = $w, θx = $θx, θy = $θy"
        )
        break
    end
end

ux_vals = Float64[]
uy_vals = Float64[]
w_vals  = Float64[]
θx_vals = Float64[]
θy_vals = Float64[]

for n in mid_nodes
    for cell in CellIterator(dh)
        idx = findfirst(==(n), cell.nodes)
        idx === nothing && continue

        cdofs = celldofs(cell)

        push!(ux_vals, u[cdofs[3*(idx-1) + 1]])
        push!(uy_vals, u[cdofs[3*(idx-1) + 2]])
        push!(w_vals,  u[cdofs[3*(idx-1) + 3]])
        push!(θx_vals, u[cdofs[9 + 2*(idx-1) + 1]])
        push!(θy_vals, u[cdofs[9 + 2*(idx-1) + 2]])

        break
    end
end

println("Average : ux = $(mean(ux_vals)), uy = $(mean(uy_vals)), w = $(mean(w_vals)), θx = $(mean(θx_vals)), θy = $(mean(θy_vals))")

println("Maximum : ux = $(maximum(abs, ux_vals)), uy = $(maximum(abs, uy_vals)), w = $(maximum(abs, w_vals)), θx = $(maximum(abs, θx_vals)), θy = $(maximum(abs, θy_vals))")