using Ferrite, Tensors, TriShellFiniteElement, MKL
    

t = 1.0

p = 1.0 # N/mm²              # Uniform pressure load intensity
E = 200000.0 #MPa
ν = 0.30   



Db = TriShellFiniteElement.calculate_bending_constitutive_matrix(E, ν, t)






struct IP3 <: ScalarInterpolation{RefTriangle, 2}
end

function Ferrite.reference_shape_value(ip::IP3, ξ::Vec{2}, shape_number::Int)
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]

    shape_number == 1 && return 1 - ξ₁ - ξ₂    
    shape_number == 2 && return ξ₁
    shape_number == 3 && return ξ₂ 

    throw(ArgumentError("no shape function $shape_number for interpolation $ip"))
end
   

Ferrite.getnbasefunctions(::IP3) = 3

Ferrite.adjust_dofs_during_distribution(::IP3) = false

##############


ip3 = TriShellFiniteElement.IP3()
qr1 = QuadratureRule{RefTriangle}(1)  

    
ip_shape = ip3 


  cv = CellValues(qr1, ip3, ip3) 



#######


  P1a = [0.0, 0.0]
  P2a = [100.0, 0.0]
  P3a = [0.0, 1000.0]

    cell_local = [P1a, P2a, P3a]

    x = [Tensors.Vec((cell_local[i][1], cell_local[i][2])) for i in eachindex(cell_local)]



    reinit!(cv, x)
    

# ke_membrane = TriShellFiniteElement.calculate_element_membrane_stiffness_matrix(Dm, cv)

   ke_b = TriShellFiniteElement.calculate_element_bending_stiffness_matrix(Db, cv)

  ind = [1:10; 13; 16]
  ke_b[ind, ind]

    
    qr = qr1

    num_shape_functions = getnbasefunctions(ip_shape)

    println("num_shape_functions:", num_shape_functions)

    ke = zeros(Float64, 6, 6)
    

    # for q_point in 1:getnquadpoints(cv)

        q_point = 1

        ξ = qr.points[q_point]

        #what is the 1?
        J = Ferrite.getjacobian(Ferrite.calculate_mapping(cv.geo_mapping, 1, x))

        Jinv = Ferrite.calculate_Jinv(Ferrite.getjacobian(Ferrite.calculate_mapping(cv.geo_mapping, 1, x)))


        Ferrite.getdetJdV(cv, q_point) 

         println("ξ:", ξ)


        # J = get_jacobian(ξ, ip_geo, x)
        Jinv = inv(J)


        # J_inv = Ferrite.getinvJ(cv, q_point) 

          println("Jinv:", Jinv)

        # nnodes = getnbasefunctions(cv)
        # dNdx = [shape_gradient(cv, q_point, i) for i in 1:nnodes]

       
        B_node_all = []
    
        dNdx = cv.fun_values.dNdx

        for i in 1:3  
       
            # i = 1

            #this is the derivative of the shape function with respect to ξ in the isoparametric element coordinate system 
            # dNdξ1 = cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][1]
            
            # dNdξ2 = cv.fun_values.dNdξ[i + (q_point-1)*num_shape_functions][2]


            # dNdx1 = cv.fun_values.dNdx[i + (q_point-1)*num_shape_functions][1]

            # dNdx2 = cv.fun_values.dNdx[i + (q_point-1)*num_shape_functions][2]


            #this is chain rule, dN/dξ * dξ/dx 
            # B_node = [dNdξ1*Jinv[1,1] + dNdξ2*Jinv[1,2]         0.0
            #             0.0                                       dNdξ1*Jinv[2,1] + dNdξ2*Jinv[2,2]
            #             dNdξ1*Jinv[2,1] + dNdξ2*Jinv[2,2]         dNdξ1*Jinv[1,1] + dNdξ2*Jinv[1,2]]
            
            B_node = [dNdx[i][1]  0.0
                      0.0         dNdx[i][2]
                      dNdx[i][2]  dNdx[i][1]]            

            push!(B_node_all, B_node)

        end


            dΩ    = getdetJdV(cv, q_point) 

        B = hcat(B_node_all...)

        println("B:", B)
        println("qr.weights[q_point]:", qr.weights[q_point])

        D = Dm 
    
        ke += (B' * D) * B .* det(J) .* qr.weights[q_point]

        # ke += B' * D * B .* getdetJdV(cv, q_point) 

      # end 

        println("ke:", ke)
