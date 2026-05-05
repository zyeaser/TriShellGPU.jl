using TriShellFiniteElement
using Ferrite



num_elem_transverse = [1]
num_elem_long = [1]
nel = (num_elem_transverse[1], num_elem_long[1])

X0_Y0  = Ferrite.Vec(0.0, 0.0)
XL_YL = Ferrite.Vec(100.0, 1000.0)
grid = generate_grid(Ferrite.Triangle, nel, X0_Y0, XL_YL)


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

    cv = CellValues(qr1, ip3, ip3) 
    

    first(CellIterator(dh))

for cell in CellIterator(dh)
    reinit!(cv, cell)
    for q_point in 1:getnquadpoints(cv)
        println("q_point", q_point)
        J     = getJ(cv, q_point)        # Jacobian matrix ∂x/∂ξ
        println("J",J)
        J_inv = getinvJ(cv, q_point)     # Inverse Jacobian ∂ξ/∂x
        dΩ    = getdetJdV(cv, q_point)   # det(J) × weight (for integration)
        
        @show J, det(J), dΩ
    end
end