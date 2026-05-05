# test_single_element.jl
#
# Verification test: GPU element stiffness ke == CPU element stiffness ke
# for a single hand-crafted triangle element.
#
# Confirms that element_stiffness_kernel! produces the same 18×18 ke as the
# CPU reference (TriShellFiniteElement.jl local_elastic_stiffness_matrix!).
# Run before trusting the full assembly pipeline.
#
# How to run:
#   julia --project=. experiments/test_single_element.jl

using CUDA
using StaticArrays
using LinearAlgebra
using Printf
using Ferrite
using Tensors

# ── Load GPU implementation ───────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "gpu", "device_functions.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu", "element_stiffness_kernel.jl"))

# ── Load CPU reference ────────────────────────────────────────────────────────
# TriShellFiniteElement is a local package at resources/TriShellFiniteElement.jl.
# It must be registered in the main project via Pkg.develop (run setup_euler.jl
# once on a new machine).  After that, plain `using` works like any other package.
using TriShellFiniteElement

# ── Element definition (one triangle in 3-D space) ────────────────────────────
#
# Three nodes in a plane tilted at 45° to exercise the rotation step.
#   Node 1: origin
#   Node 2: 1 m along X
#   Node 3: 0.5 m along X, 0.5 m along Y, 0.5 m along Z
#
coords_mat = [
    0.0   0.0   0.0 ;
    1.0   0.0   0.0 ;
    0.5   0.5   0.5
]   # (3 × 3), rows = nodes, cols = X,Y,Z

# Convert to Vector{Tensors.Vec{3,Float64}} — the format TriShellFiniteElement expects
x_global = [Tensors.Vec{3,Float64}((coords_mat[i,1], coords_mat[i,2], coords_mat[i,3]))
            for i in 1:3]

# Material
E = 2.1e11   # Pa (steel Young's modulus)
ν = 0.3
t = 0.01     # m (plate thickness)

# ── CPU reference ke ──────────────────────────────────────────────────────────
#
# Follows the exact same pipeline used in TriShellFiniteElement.assemble_global_Ke!:
#   1. Build local coordinate frame (rotation matrix T, 3×3)
#   2. Project global 3-D coords to planar 2-D local coords
#   3. Build Ferrite quadrature rules and interpolations
#   4. Compute 18×18 ke in local frame
#   5. Rotate to global frame
#   6. Reorder DOFs from component-by-node to field-by-field (Ferrite convention)

# Step 1: rotation matrix
T = TriShellFiniteElement.calculation_rotation_matrix(x_global)  # 3×3

# Step 2: local 2-D coordinates (Vector of Vec{2})
x_local = TriShellFiniteElement.global_nodal_coords_to_planar_coords(x_global, T)

# Step 3: quadrature rules and interpolations
qr1 = QuadratureRule{RefTriangle}(1)      # 1-point rule for membrane/bending
qr3 = QuadratureRule{RefTriangle}(3)      # degree-3 rule (4 pts) for shear
ip3 = TriShellFiniteElement.IP3()         # linear (3-node) interpolation
ip6 = TriShellFiniteElement.IP6()         # quadratic (6-node) interpolation

# Step 4: local stiffness matrix (18×18, local frame)
ke_local = TriShellFiniteElement.local_elastic_stiffness_matrix!(
    qr1, qr3, ip3, ip6, E, ν, t, x_local)

# Step 5: rotate to global frame
Te = TriShellFiniteElement.rotation_matrix_for_element_stiffness_drilling(T)
ke_global_node = Te * ke_local * Te'

# Step 6: field-by-field reordering (Ferrite convention)
ind_field = [1, 2, 3, 7, 8, 9, 13, 14, 15, 4, 5, 6, 10, 11, 12, 16, 17, 18]
ke_cpu = ke_global_node[ind_field, ind_field]

# ── GPU ke ────────────────────────────────────────────────────────────────────
#
# Pack coords into (n_elem=1, n_node=3, xyz=3) for the kernel.
# kernel signature: coords[elem, node, xyz]
coords_gpu = Array{Float64,3}(undef, 1, 3, 3)
for nd in 1:3, xyz in 1:3
    coords_gpu[1, nd, xyz] = coords_mat[nd, xyz]
end

coords_d  = CuArray(coords_gpu)
ke_all_d  = CUDA.zeros(Float64, 1, 18, 18)

@cuda threads=1 blocks=1 element_stiffness_kernel!(
    ke_all_d, coords_d, E, ν, t, Int32(1))
CUDA.synchronize()

ke_gpu = Array(ke_all_d)[1, :, :]   # (18 × 18) on CPU

# ── Comparison ────────────────────────────────────────────────────────────────
abs_err = abs.(ke_cpu .- ke_gpu)
max_abs = maximum(abs_err)
max_ref = maximum(abs.(ke_cpu))
rel_err = max_abs / (max_ref + 1e-300)

sym_cpu = maximum(abs.(ke_cpu .- ke_cpu'))
sym_gpu = maximum(abs.(ke_gpu .- ke_gpu'))

println("─────────────────────────────────────────")
println("Single-element GPU vs CPU ke comparison")
println("─────────────────────────────────────────")
@printf("  max |ke_cpu - ke_gpu|  = %.6e\n", max_abs)
@printf("  max |ke_cpu|           = %.6e\n", max_ref)
@printf("  relative error         = %.6e\n", rel_err)
println()
@printf("  symmetry error (CPU ke) = %.6e\n", sym_cpu)
@printf("  symmetry error (GPU ke) = %.6e\n", sym_gpu)
println()

tol_abs = 1e-6
tol_rel = 1e-8

if max_abs < tol_abs && rel_err < tol_rel
    println("PASS: ke_gpu matches ke_cpu within tolerance")
else
    println("FAIL: maximum error $max_abs exceeds tolerance $tol_abs")
    println()
    println("Top-10 worst entries (row, col, cpu_val, gpu_val, abs_err):")
    flat_err = sort([(abs_err[i,j], i, j) for i in 1:18, j in 1:18][:]; rev=true)
    for (e, r, c) in flat_err[1:min(10,length(flat_err))]
        @printf("  (%2d,%2d)  cpu=%12.4e  gpu=%12.4e  err=%10.3e\n",
                r, c, ke_cpu[r,c], ke_gpu[r,c], e)
    end
end
