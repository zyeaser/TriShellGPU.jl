

using Ferrite, TriShellFiniteElement 


E = 200000.0
t = 1.0
ν = 0.30

######
grid = generate_grid(Triangle, (1, 1), Vec((0.0, 0.0)), Vec((2.0, 1.0)));                      

ip = Lagrange{RefTriangle,1}() #to define fields only 
ip3 = TriShellFiniteElement.IP3()
qr1 = QuadratureRule{RefTriangle}(1)  

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)


ip_shape = ip3
ip_geo = ip3

qr = qr1

cv = CellValues(qr1, ip3, ip3) 


cell = first(CellIterator(dh))

cells = collect(CellIterator(dh))

cell = cells[2]

x = getcoordinates(cell)

x3 = [Vec{3, Float64}([x[i]; 0.0]) for i in eachindex(x)]

P1 = x3[1]
P2 = x3[2]
P3 = x3[3]

using LinearAlgebra


    norm_vec=cross(P2-P1,P3-P1);
    norm_vec=norm_vec/norm(norm_vec);
    j3=norm_vec;
    j1=(P2-P1)/norm(P2-P1)
    j2=cross(j3,j1)
    T=[j1 j2 j3]


P1a=T'*(P1-P1)
P2a=T'*(P2-P1)
P3a=T'*(P3-P1)



str_mat_global = [0.0 0.0
                  0.0 1.0]

str_mat_local =  T[1:2,1:2]' * str_mat_global * T[1:2, 1:2]          




    str_mat(1,1)=strxx;str_mat(2,2)=stryy;str_mat(1,2)=strxy;str_mat(2,1)=strxy;
    str_mat=T(1:2,1:2)'*str_mat*T(1:2,1:2);
    strxx=str_mat(1,1);stryy=str_mat(2,2);strxy=str_mat(1,2);
