using Ferrite, Tensors, TriShellFiniteElement, MKL
    

t = 1.0

p = 1.0 # N/mm²              # Uniform pressure load intensity
E = 200000.0 #MPa
ν = 0.30   





    Ds = TriShellFiniteElement.calculate_shear_constitutive_matrix(E, ν, t)




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

struct IP6 <: ScalarInterpolation{RefTriangle, 2}
end


function Ferrite.reference_shape_value(ip::IP6, ξ::Vec{2}, shape_number::Int)
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]

    shape_number == 1 && return 1 - ξ₁ - ξ₂
    shape_number == 2 && return ξ₁
    shape_number == 3 && return ξ₂
    shape_number == 4 && return 4* ξ₁ * (1 - ξ₁ - ξ₂)
    shape_number == 5 && return 4 * ξ₁ * ξ₂
    shape_number == 6 && return 4 * ξ₂ * (1 - ξ₁ - ξ₂)

    throw(ArgumentError("no shape function $shape_number for interpolation $ip"))
end


Ferrite.getnbasefunctions(::IP6) = 6

Ferrite.adjust_dofs_during_distribution(::IP6) = false





ip3 = TriShellFiniteElement.IP3()
ip6 = TriShellFiniteElement.IP6()
# qr1 = QuadratureRule{RefTriangle}(1)  

qr3 = QuadratureRule{RefTriangle}(2)  
    
# ip_shape = ip3 


cv = CellValues(qr3, ip6, ip3)
  

#######


  P1a = [0.0, 0.0]
  P2a = [100.0, 0.0]
  P3a = [0.0, 1000.0]

    cell_local = [P1a, P2a, P3a]

    x = [Tensors.Vec((cell_local[i][1], cell_local[i][2])) for i in eachindex(cell_local)]



    reinit!(cv, x)
    
dNdx = cv.fun_values.dNdx


# ke_membrane = TriShellFiniteElement.calculate_element_membrane_stiffness_matrix(Dm, cv)

ke_shear = TriShellFiniteElement.calculate_element_shear_stiffness_matrix(Ds, cv)

  # ind = [1:10; 13; 16]
  # ke_shear[ind, ind]

    