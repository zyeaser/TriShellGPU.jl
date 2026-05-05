using TriShellFiniteElement
using Ferrite
# using Arpack
using LinearAlgebra
using CairoMakie
t = 2.0

p = 1.0 # N/mm²              # Uniform pressure load intensity
E = 200000.0 #MPa
ν = 0.30   

num_elem_transverse = [2]
num_elem_long = [8]
nel = (num_elem_transverse[1], num_elem_long[1])

X0_Y0  = Ferrite.Vec(0.0, 0.0)
XL_YL = Ferrite.Vec(100.0, 1000.0)
grid = generate_grid(Ferrite.Triangle, nel, X0_Y0, XL_YL)

grid.nodes

nodes_3D = [Node((grid.nodes[i].x[1], grid.nodes[i].x[2], 0.0)) for i in eachindex(grid.nodes)]

grid = Grid(grid.cells, nodes_3D)

# cell = first(CellIterator(dh))

# x = getcoordinates(cell)


ip = Lagrange{RefTriangle,1}()
ip6 = TriShellFiniteElement.IP6()
ip3 = TriShellFiniteElement.IP3()
qr1 = QuadratureRule{RefTriangle}(1)  
qr3 = QuadratureRule{RefTriangle}(2)  


dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

Ke = allocate_matrix(dh)
Ke = TriShellFiniteElement.assemble_global_Ke!(Ke, dh, qr1, qr3, ip3, ip6, E, ν, t)

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

    if     y == XL_YL[2]/2  && x == XL_YL[1]/2 
        push!(middle_node, i)
    elseif y == X0_Y0[2]   && x == X0_Y0[1] 
        push!(left_top, i)
    elseif y == XL_YL[2] && x == X0_Y0[1]
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




ch = ConstraintHandler(dh)
# uy restrained at middle node
add!(ch, Dirichlet(:u, middle_node,  (x, t) -> [0.0], [2]))

# ux restrained at top corner nodes
add!(ch, Dirichlet(:u, left_top,  (x, t) -> [0.0], [1]))
add!(ch, Dirichlet(:u, right_top, (x, t) -> [0.0], [1]))
add!(ch, Dirichlet(:u, left_nodes,  (x, t) -> [0.0], [3]))
add!(ch, Dirichlet(:u, right_nodes, (x, t) -> [0.0], [3]))

close!(ch)

apply!(Ke, ch)

# interior_Y0  = Int[]
# interior_YL  = Int[]
# corner_X0_Y0 = Int[]
# corner_X0_YL = Int[]
# corner_XL_Y0 = Int[]
# corner_XL_YL = Int[]


# tol = 1e-8

# for (i, node) in enumerate(grid.nodes)
#     x, y = node.x

#     if     y == X0_Y0[2] && abs(x - XL_YL[1]/2) < tol
#         push!(interior_Y0, i)
#     elseif y == XL_YL[2] && abs(x - XL_YL[1]/2) < tol
#         push!(interior_YL, i)
#     elseif y == X0_Y0[2] && x == X0_Y0[1]
#         push!(corner_X0_Y0, i)
#     elseif y == XL_YL[2] && x == X0_Y0[1]
#         push!(corner_X0_YL, i)
#     elseif y == X0_Y0[2] && x == XL_YL[1]
#         push!(corner_XL_Y0, i)
#     elseif y == XL_YL[2] && x == XL_YL[1]
#         push!(corner_XL_YL, i)
#     end
# end



# for node in left_nodes
#     dofs = node_to_dofs[node]
#     F[dofs[2]] -= p * right[1] / length(left_nodes)
# end
# for node in right_nodes
#     dofs = node_to_dofs[node]
#     F[dofs[2]] += p * right[1] / length(right_nodes)
# end

# load_magnitude = (1.0 * t * XL_YL[1]) / length(left_nodes)

# apply!(F, ch)


# u = Ke \ F
# apply!(u, ch)

#assume a negative stress is compression
σ_elem = Vector{Ferrite.Vec{3,Float64}}()
for cell in CellIterator(dh)
    push!(σ_elem, Ferrite.Vec(0.0, -p, 0.0))
end


# cell = first(CellIterator(dh))

# x_global = getcoordinates(cell)

# T = TriShellFiniteElement.calculation_rotation_matrix(x_global)

# P1 = x_global[1]
#     P2 = x_global[2]
#     P3 = x_global[3]

#     P1a=T'*(P1-P1)
#     P2a=T'*(P2-P1)
#     P3a=T'*(P3-P1)

#     # nodes_local = [P1a, P2a, P3a]

#         cell_local = [P1a, P2a, P3a]

#     cell_nodes_local = [Node((cell_local[i][1], cell_local[i][2])) for i in eachindex(cell_local)]

#     using Tensors
#     x_local = [Tensors.Vec((nodes_local[i][1], nodes_local[i][2])) for i in eachindex(nodes_local)]




Kg = allocate_matrix(dh)
Kg = TriShellFiniteElement.assemble_global_Kg!(Kg, dh, qr1, ip3, σ_elem)
apply!(Kg, ch)

# result = eigen(Matrix(Ke), -Matrix(Kg))
# eigenvalues = result.values
# eigenvectors = result.vectors

eigenvalues = eigvals(Matrix(Ke), -Matrix(Kg))   
eigenvectors = eigvecs(Matrix(Ke), -Matrix(Kg))


# Filter to finite positive eigenvalues and sort them
pos_indices = findall(v -> isfinite(v) && v > 0, eigenvalues)
sorted_indices = pos_indices[sortperm(eigenvalues[pos_indices])]

n_modes = min(2, length(sorted_indices))  # number of modes to extract
println("First $n_modes buckling eigenvalues (load factors):")
for i in 1:n_modes
    idx = sorted_indices[i]
    println("  Mode $i: λ = $(eigenvalues[idx])")
end 



########################################################
######<<<<<<<<<<<<  Visualization >>>>>>>>>>>>>>########
########################################################

using CairoMakie

mode1_idx = sorted_indices[1]
mode1_vector = eigenvectors[:, mode1_idx]

# Normalize mode shape (important!)
mode1_vector ./= maximum(abs.(mode1_vector))

n_nodes = length(grid.nodes)
u_node = zeros(n_nodes, 3)   # we only need ux, uy, w

for cell in CellIterator(dh)
    cdofs = celldofs(cell)

    for (i, n) in enumerate(cell.nodes)
        # 5 dofs per node
        base = 5*(i-1)

        u_node[n,1] = mode1_vector[cdofs[base + 1]]  # ux
        u_node[n,2] = mode1_vector[cdofs[base + 2]]  # uy
        u_node[n,3] = mode1_vector[cdofs[base + 3]]  # w
    end
end

using GeometryBasics
faces = GeometryBasics.TriangleFace{Int}[]

for cell in grid.cells
    push!(faces, GeometryBasics.TriangleFace(cell.nodes...))
end

scale = 20.0 

vertices = Point3f[]

for i in 1:n_nodes
    x = grid.nodes[i].x[1] + scale * u_node[i,1]
    y = grid.nodes[i].x[2] + scale * u_node[i,2]
    z = scale * u_node[i,3]

    push!(vertices, Point3f(x, y, z))
end

gb_mesh = GeometryBasics.Mesh(vertices, faces)

fig = Figure(size = (900,700))

ax = Axis3(
    fig[1,1],
    xlabel = "X (mm)",
    ylabel = "Y (mm)",
    zlabel = "Z (mm)",
    title = "Buckling Mode Shape",
    aspect = :data
)

mesh!(ax, gb_mesh,
      color = u_node[:,3],         
      colormap = :turbo,
      colorrange = (-1, 1))

wireframe!(ax, gb_mesh,
           color = :black,
           linewidth = 0.5)

Colorbar(fig[1,2],
         colormap = :turbo,
         label = "Normalized w")

fig
