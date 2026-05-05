# geometric_stiffness_kernel.jl — Step 6 (Buckling Extension)
#
# One CUDA thread per element.
# Each thread computes the 18×18 geometric stiffness matrix kg for its element
# using the pre-computed element stresses, and stores it in kg_all.
#
# This is the GPU counterpart of CPU reference functions:
#   calculate_element_geometric_stiffness_matrix()  — src/TriShellFiniteElement.jl:452
#   assemble_global_Kg!()                           — src/TriShellFiniteElement.jl:582
#
# Kernel signature (planned):
#   @cuda threads=... blocks=... geometric_stiffness_kernel!(
#       kg_all,        # CuArray{Float64, 3}  shape: (n_elem, 18, 18)
#       coords_all,    # CuArray{Float64, 3}  shape: (n_elem, 3, 3) — node XYZ per element
#       sigma_all,     # CuArray{Float64, 2}  shape: (n_elem, 3)    — [σxx, σyy, τxy] global
#       n_elem
#   )
#
# ── Per-thread steps ──────────────────────────────────────────────────────────
#
#   1. Load node coordinates P1, P2, P3 and stresses [σXX, σYY, τXY] for this element
#
#   2. Compute local frame and rotation matrix (same as Step 2 in element_stiffness_kernel):
#        R = local_to_global_rotation(P1, P2, P3)   →  3×3
#        T = expand_rotation_18(R)                   → 18×18
#
#   3. Project node coords to local 2D:
#        x_local = project_to_local(P1, P2, P3, R)   →  3×2
#
#   4. Transform element stresses from global to local coordinate system:
#      The CPU reference uses a 2D rotation via the T2 matrix (lines 548-551):
#        c = R[1,1],  s = R[2,1]
#        T2 = [c^2  s^2   2sc;   s^2  c^2  -2sc;   -sc  sc  c^2-s^2]
#        str_vec_local = T2 * [σXX, σYY, τXY]
#        → σxx_local, σyy_local, τxy_local  (in local element frame)
#
#   5. Compute physical shape function derivatives (IP3, same quadrature as Step 2):
#      For each quadrature point (xi_q, eta_q, w_q) using qr1 or qr3:
#        N_ip3, dN_dxi, dN_deta = shape_functions_ip3(xi_q, eta_q)
#        J, Jinv, detJ = compute_jacobian(x_local, dN_dxi, dN_deta)
#        dN_dx = Jinv * dN_dxi      (length 3)
#        dN_dy = Jinv * dN_deta
#
#   6. Build geometric B matrices for in-plane (uv) and out-of-plane (wθ) DOFs:
#
#      For membrane (u,v) — 2×6 matrix dNuv_x and dNuv_y:
#        dNuv_x = [dN1/dx  0  dN2/dx  0  dN3/dx  0
#                   0  dN1/dx   0  dN2/dx   0  dN3/dx]
#        dNuv_y = [dN1/dy  0  dN2/dy  0  dN3/dy  0
#                   0  dN1/dy   0  dN2/dy   0  dN3/dy]
#        GGuvx  = dNuv_x' * dNuv_x                    →  6×6
#        GGuvy  = dNuv_y' * dNuv_y                    →  6×6
#        GGuvxy = dNuv_x' * dNuv_y + dNuv_y' * dNuv_x →  6×6
#
#      For bending+shear (w,θx,θy) — 3×9 matrix dNwt_x and dNwt_y:
#        Only the w-DOFs contribute (columns 1, 4, 7 of the 9-DOF block):
#        dNwt_x[1, 1] = dN1/dx;  dNwt_x[1, 4] = dN2/dx;  dNwt_x[1, 7] = dN3/dx
#        dNwt_y[1, 1] = dN1/dy;  dNwt_y[1, 4] = dN2/dy;  dNwt_y[1, 7] = dN3/dy
#        (all other entries zero)
#        GGwtx  = dNwt_x' * dNwt_x                    →  9×9
#        GGwty  = dNwt_y' * dNwt_y                    →  9×9
#        GGwtxy = dNwt_x' * dNwt_y + dNwt_y' * dNwt_x →  9×9
#
#   7. Accumulate geometric stiffness contributions:
#        kuv += (σxx_local*GGuvx + σyy_local*GGuvy + τxy_local*GGuvxy) * detJ * w_q
#        kwt += (σxx_local*GGwtx + σyy_local*GGwty + τxy_local*GGwtxy) * detJ * w_q
#
#   8. Assemble 15×15 local kg from kuv and kwt blocks:
#        induv = [1,2, 6,7, 11,12]              — u,v DOF indices in 15-DOF layout
#        indwt = [3,4,5, 8,9,10, 13,14,15]      — w,θx,θy DOF indices
#        kg_local_15[induv, induv] = kuv
#        kg_local_15[indwt, indwt] = kwt
#      Matches CPU reference: assemble_global_Kg! lines 561-563
#
#   9. Expand 15×15 → 18×18 (add drilling DOF rows/cols as zero):
#        ind = [1:5; 7:11; 13:17]               — maps 15-DOF to 18-DOF positions
#        kg_local_18 = zeros(18, 18)
#        kg_local_18[ind, ind] = kg_local_15
#
#  10. Rotate to global:
#        kg_global = T * kg_local_18 * T'
#
#  11. Reorder DOFs for Ferrite global assembly (same as ke in Step 2):
#        ind_field = [1,2,3, 7,8,9, 13,14,15, 4,5,6, 10,11,12, 16,17,18]
#        kg_global = kg_global[ind_field, ind_field]
#
#  12. Store kg_all[elem_id, :, :] = kg_global
#
# ── Notes ─────────────────────────────────────────────────────────────────────
#
#   - Stresses are assumed constant per element (computed from the linear solve).
#     The CPU reference stores one (σxx, σyy, τxy) triple per element.
#   - The CPU reference's assemble_global_Kg! has a bug (lines 605-607):
#       σxx_element, σyy_element, τxy_element are referenced but never assigned.
#       Fix: replace with σXX_element, σYY_element, τXY_element (the uppercase ones).
#   - After assembly, the global Kg shares the same CSR sparsity as K (same mesh).
#     The assembly_kernel! from Step 3 can be reused with kg_all and a zeroed Kg_values.
#   - Eigenvalue solve (K * phi = lambda * Kg * phi) is done on CPU via Arpack.jl
#     or on GPU via CUSOLVER if available. See gpu_solve.jl Step E.
#
