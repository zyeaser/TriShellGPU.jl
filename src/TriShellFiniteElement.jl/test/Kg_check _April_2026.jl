using TriShellFiniteElement
using Ferrite
# using Arpack
using LinearAlgebra
using CairoMakie
t = 1.0

p = 1.0 # N/mm²              # Uniform pressure load intensity
E = 200000.0 #MPa
ν = 0.30   

num_elem_transverse = [1]
num_elem_long = [1]
nel = (num_elem_transverse[1], num_elem_long[1])

X0_Y0  = Ferrite.Vec(0.0, 0.0)
XL_YL = Ferrite.Vec(100.0, 1000.0)
grid = generate_grid(Ferrite.Triangle, nel, X0_Y0, XL_YL)


nodes_3D = [Node((grid.nodes[i].x[1], grid.nodes[i].x[2], 0.0)) for i in eachindex(grid.nodes)]

grid = Grid(grid.cells, nodes_3D)


ip = Lagrange{RefTriangle,1}()
ip3 = TriShellFiniteElement.IP3()
qr1 = QuadratureRule{RefTriangle}(1)  

cv = CellValues(qr1, ip3, ip3)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)




# strxx = [0.0, 0.0]
# stryy = [1.0, 1.0]
# strxy = [0.0, 0.0]


cell = first(CellIterator(dh))


 x_global = getcoordinates(cell)


T = TriShellFiniteElement.calculation_rotation_matrix(x_global)


x_local = TriShellFiniteElement.global_nodal_coords_to_planar_coords(x_global, T)



σxx_element = [0.0]
σyy_element = [1.0]
τxy_element = [0.0]


reinit!(cv, x_local)


kg = TriShellFiniteElement.calculate_element_geometric_stiffness_matrix(cv, σxx_element, σyy_element, τxy_element, T)

# TriShellFiniteElement.calculate_element_geometric_stiffness_matrix(cv, σ)



# dh = DofHandler(grid)
# add!(dh, :u, ip^3)
# add!(dh, :θ, ip^3)
# close!(dh)

# Ke = allocate_matrix(dh)
# Ke = TriShellFiniteElement.assemble_global_Ke!(Ke, dh, qr1, qr3, ip3, ip6, E, ν, t)


# index_matlab = [1, 2, 3, 13, 14, 4, 5, 6, 15, 16, 7, 8, 9, 17, 18, 10, 11, 12, 19, 20]


# Ke_matlab = Matrix(Ke[index_matlab, index_matlab])

