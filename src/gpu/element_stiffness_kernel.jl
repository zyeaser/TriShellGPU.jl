# element_stiffness_kernel.jl — Step 2
#
# One CUDA thread per element.
# Each thread computes the full 18×18 local→global element stiffness matrix ke
# for its assigned element and stores it in ke_all.
#
# Kernel signature (planned):
#   @cuda threads=... blocks=... element_stiffness_kernel!(
#       ke_all,        # CuArray{Float64, 3}  shape: (n_elem, 18, 18)
#       coords_all,    # CuArray{Float64, 3}  shape: (n_elem, 3, 3) — node XYZ per element
#       E, nu, t       # scalar material properties
#   )
#
# ── Per-thread steps ──────────────────────────────────────────────────────────
#
#   1. Load node coordinates P1, P2, P3 (global XYZ) for this element
#      from coords_all[elem_id, :, :]
#
#   2. Compute local coordinate system via device_functions.jl:
#        R   = local_to_global_rotation(P1, P2, P3)     →  3×3
#        T   = expand_rotation_18(R)                     → 18×18 block-diagonal
#
#   3. Project global node coords into local 2D frame:
#        x_local = project_to_local(P1, P2, P3, R)       →  3×2  (x̄, ȳ only)
#      For the shear IP6 field, also compute midside node coords:
#        x_mid[1] = (x_local[1] + x_local[2]) / 2       — midpoint of edge 1-2
#        x_mid[2] = (x_local[2] + x_local[3]) / 2       — midpoint of edge 2-3
#        x_mid[3] = (x_local[3] + x_local[1]) / 2       — midpoint of edge 3-1
#        x_local_ip6 = vcat(x_local, x_mid)              →  6×2
#
#   4. Constitutive matrices (all constant per element, computed once):
#        G  = E / (2*(1+nu))
#        Dm = (E/(1-nu^2)) * [1  nu  0;  nu  1  0;  0  0  (1-nu)/2]  * t
#        Db = (E*t^3/12/(1-nu^2)) * [1  nu  0;  nu  1  0;  0  0  (1-nu)/2]
#        Ds = (5/6*G*t) * I(2)
#      NOTE: Dm is already scaled by t; do NOT multiply again in the quadrature loop.
#            This matches CPU reference: calculate_membrane_constitutive_matrix()
#
#   5. Membrane stiffness km (6×6) — IP3, 1-point or 3-point Gauss rule:
#      For each quadrature point (xi_q, eta_q, w_q):
#        N_ip3, dN_dxi_ip3, dN_deta_ip3 = shape_functions_ip3(xi_q, eta_q)
#        J, Jinv, detJ = compute_jacobian(x_local, dN_dxi_ip3, dN_deta_ip3)
#        dN_dx = Jinv[1,1]*dN_dxi + Jinv[1,2]*dN_deta   (IP3, length 3)
#        dN_dy = Jinv[2,1]*dN_dxi + Jinv[2,2]*dN_deta
#        B = B_membrane(dN_dx, dN_dy)                    →  3×6
#        km += B' * Dm * B * detJ * w_q
#
#   6. Bending stiffness kb (9×9) — IP3, same quadrature:
#      For each quadrature point:
#        (reuse dN_dx, dN_dy from step 5)
#        B = B_bending(dN_dx, dN_dy)                     →  3×9
#        kb += B' * Db * B * detJ * w_q
#
#   7. Shear stiffness ks (12×12) — IP6, 3-point Gauss rule (qr3 in CPU ref):
#      The shear field has 12 DOFs: [w1,θx1,θy1, w2,θx2,θy2, w3,θx3,θy3, w4,w5,w6]
#      where w4,w5,w6 are midside w-DOFs (internal, condensed out in step 8).
#      For each quadrature point:
#        N_ip3_q, _, _ = shape_functions_ip3(xi_q, eta_q)       (for θ terms)
#        N_ip6_q, dN_dxi_ip6, dN_deta_ip6 = shape_functions_ip6(xi_q, eta_q)
#        J6, Jinv6, detJ6 = compute_jacobian(x_local_ip6, dN_dxi_ip6, dN_deta_ip6)
#        dN_dx_ip6 = Jinv6 * dN_dxi_ip6    (length 6)
#        dN_dy_ip6 = Jinv6 * dN_deta_ip6
#        B = B_shear(N_ip3_q, dN_dx_ip6, dN_dy_ip6)            →  2×12
#        ks += B' * Ds * B * detJ6 * w_q
#
#   8. Static condensation of midside shear w-DOFs:
#      Partition ks into vertex DOFs (a = 1:9) and midside DOFs (i = 10:12):
#        ks_cond = ks[a,a] - ks[a,i] * inv(ks[i,i]) * ks[i,a]     →  9×9
#      This matches CPU reference: local_elastic_stiffness_matrix! lines 344-346.
#      NOTE: This is condensation of midside w-DOFs, NOT the drilling DOF θz.
#
#   9. Adaptive shear correction:
#      Compute correction factor from ratio of shear-to-bending rotational stiffness:
#        alpha = sum(diag(ks_cond[4:9, 4:9])) / sum(diag(kb[4:9, 4:9]))
#        Cs    = 0.0   (correction coefficient; set to 0 to skip blending)
#        ke_bs = (1 / (1 + Cs*alpha)) * ks_cond + kb               →  9×9
#      NOTE: This matches CPU reference exactly. Using a fixed k=5/6 in Ds (step 4)
#            instead is simpler but will produce slightly different results.
#            Recommend matching the CPU reference adaptive alpha for validation.
#
#  10. Drilling DOF penalty:
#      The 18-DOF element has in-plane rotation DOFs θz at nodes 1,2,3 (indices 6,12,18).
#      These have no physical stiffness; add a small penalty to prevent singularity:
#        stif = min(diag(ke_local)[rotation_indices]) / 100
#        ke_local[6,6]   = stif
#        ke_local[12,12] = stif
#        ke_local[18,18] = stif
#      Matches CPU reference: local_elastic_stiffness_matrix! lines 378-383.
#
#  11. Assemble 18×18 local stiffness ke_local from blocks:
#      DOF layout (local element, before global rotation):
#        Per node: [u, v, w, θx, θy, θz]  →  nodes 1,2,3  →  18 total
#      Block placement (1-indexed, same as CPU reference):
#        induv = [1,2, 7,8, 13,14]         — membrane (u,v) DOFs
#        indwt = [3,4,5, 9,10,11, 15,16,17] — bending+shear (w,θx,θy) DOFs
#        ke_local[induv, induv] = km        (6×6)
#        ke_local[indwt, indwt] = ke_bs    (9×9)
#        apply drilling penalty at [6,6], [12,12], [18,18]
#
#  12. Rotate to global:
#        ke_global = T * ke_local * T'
#      where T is the 18×18 block-diagonal from expand_rotation_18(R).
#
#  13. Reorder DOFs for Ferrite global assembly:
#      Ferrite numbers DOFs field-by-field (all u, then all v, ...) rather than
#      node-by-node. Apply reordering before storing:
#        ind_field = [1,2,3, 7,8,9, 13,14,15, 4,5,6, 10,11,12, 16,17,18]
#        ke_global = ke_global[ind_field, ind_field]
#      Matches CPU reference: assemble_global_Ke! line 429.
#
#  14. Store ke_all[elem_id, :, :] = ke_global
#
# ── Quadrature points (standard triangle Gauss rules) ─────────────────────────
#
#   qr1 (1-point):  (1/3, 1/3),  w = 1/2
#       → QR1_* in device_functions.jl; matches Ferrite QuadratureRule{RefTriangle}(1)
#   qr3 (4-point):  Ferrite QuadratureRule{RefTriangle}(3), degree-3 Dunavant rule
#       points  = [(1/3,1/3), (0.2,0.2), (0.2,0.6), (0.6,0.2)]
#       weights = [-0.28125, 0.260416…, 0.260416…, 0.260416…]
#       → QR3_* in device_functions.jl; 4 points, one negative weight (valid)
#   Use qr1 for membrane and bending; use qr3 for shear (IP6).
#   CPU reference uses qr1 for membrane/bending, qr3 for shear — match this exactly.
#

using CUDA
using StaticArrays

# ── DOF index maps (direct to 18-DOF node-by-node layout) ─────────────────────
#
# 18-DOF node-by-node layout:
#   Node 1: u1(1)  v1(2)  w1(3)  θx1(4)  θy1(5)  θz1(6)   ← θz = drilling
#   Node 2: u2(7)  v2(8)  w2(9)  θx2(10) θy2(11) θz2(12)
#   Node 3: u3(13) v3(14) w3(15) θx3(16) θy3(17) θz3(18)
#
const INDUV18 = (1, 2, 7, 8, 13, 14)                    # membrane DOFs (u,v × 3 nodes)
const INDWT18 = (3, 4, 5, 9, 10, 11, 15, 16, 17)        # bending/shear DOFs (w,θx,θy × 3 nodes)

# ── 3×3 matrix inverse (Cramer's rule, used for static condensation) ──────────
@inline function inv3x3!(B::MArray{Tuple{3,3},T,2,9},
                          A::MArray{Tuple{3,3},T,2,9}) where {T}
    a11 = A[1,1]; a12 = A[1,2]; a13 = A[1,3]
    a21 = A[2,1]; a22 = A[2,2]; a23 = A[2,3]
    a31 = A[3,1]; a32 = A[3,2]; a33 = A[3,3]
    det     =  a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
    inv_det = one(T) / det
    B[1,1] =  (a22*a33 - a23*a32) * inv_det
    B[1,2] = -(a12*a33 - a13*a32) * inv_det
    B[1,3] =  (a12*a23 - a13*a22) * inv_det
    B[2,1] = -(a21*a33 - a23*a31) * inv_det
    B[2,2] =  (a11*a33 - a13*a31) * inv_det
    B[2,3] = -(a11*a23 - a13*a21) * inv_det
    B[3,1] =  (a21*a32 - a22*a31) * inv_det
    B[3,2] = -(a11*a32 - a12*a31) * inv_det
    B[3,3] =  (a11*a22 - a12*a21) * inv_det
    return nothing
end

# ── Kernel ────────────────────────────────────────────────────────────────────
function element_stiffness_kernel!(
        ke_all    ::CuDeviceArray{Float64,3},
        coords_all::CuDeviceArray{Float64,3},
        E::Float64, ν::Float64, t::Float64,
        n_elem::Int32)

    e = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    e > n_elem && return

    # ── 1. Load global node coordinates ──────────────────────────────────────
    @inbounds begin
        P1x = coords_all[e,1,1]; P1y = coords_all[e,1,2]; P1z = coords_all[e,1,3]
        P2x = coords_all[e,2,1]; P2y = coords_all[e,2,2]; P2z = coords_all[e,2,3]
        P3x = coords_all[e,3,1]; P3y = coords_all[e,3,2]; P3z = coords_all[e,3,3]
    end

    # ── 2. Local coordinate frame ─────────────────────────────────────────────
    j1x, j1y, j1z,
    j2x, j2y, j2z,
    j3x, j3y, j3z = local_to_global_rotation(P1x, P1y, P1z,
                                               P2x, P2y, P2z,
                                               P3x, P3y, P3z)

    # ── 3. Project to local 2-D coords ───────────────────────────────────────
    lx1, ly1,
    lx2, ly2,
    lx3, ly3 = project_to_local(P1x, P1y, P1z,
                                  P2x, P2y, P2z,
                                  P3x, P3y, P3z,
                                  j1x, j1y, j1z,
                                  j2x, j2y, j2z)

    # ── 4. Jacobian (constant for a linear triangle, computed once) ───────────
    Jinv11, Jinv12, Jinv21, Jinv22, detJ =
        compute_jacobian(lx1, ly1, lx2, ly2, lx3, ly3)

    # ── 5. Constitutive matrices ──────────────────────────────────────────────
    Dm  = constitutive_membrane(E, ν, t)
    Db  = constitutive_bending(E, ν, t)
    gst = constitutive_shear_diag(E, ν, t)

    # ── 6. Stiffness accumulators ─────────────────────────────────────────────
    km = MArray{Tuple{6,6},   Float64, 2,  36}(undef); fill!(km,  0.0)
    kb = MArray{Tuple{9,9},   Float64, 2,  81}(undef); fill!(kb,  0.0)
    ks = MArray{Tuple{12,12}, Float64, 2, 144}(undef); fill!(ks,  0.0)

    # ── 7. qr1 loop — membrane + bending (1-point centroid, IP3) ─────────────
    # IP3 natural derivatives are constant → physical derivatives computed once.
    let
        _, dN_dξ₁_3, dN_dξ₂_3 = shape_functions_ip3(QR1_ξ₁[1], QR1_ξ₂[1])
        dN_dx3, dN_dy3 = phys_derivs(dN_dξ₁_3, dN_dξ₂_3, Jinv11, Jinv12, Jinv21, Jinv22)
        wdetJ1 = QR1_w[1] * detJ
        btdb_membrane!(km, B_membrane(dN_dx3, dN_dy3), Dm, wdetJ1)
        btdb_bending!( kb, B_bending( dN_dx3, dN_dy3), Db, wdetJ1)
    end

    # ── 8. qr3 loop — shear (Ferrite degree-3 rule, 4 points, IP6) ───────────
    # QR3_* has 4 entries matching Ferrite QuadratureRule{RefTriangle}(3).
    # The Jacobian is the same at every point (linear geometry), so only the
    # IP6 shape function values/derivatives change between quadrature points.
    for q in 1:4
        ξ₁q = QR3_ξ₁[q]; ξ₂q = QR3_ξ₂[q]
        N_ip3, _, _             = shape_functions_ip3(ξ₁q, ξ₂q)
        _, dN_dξ₁_6, dN_dξ₂_6 = shape_functions_ip6(ξ₁q, ξ₂q)
        dN_dx6, dN_dy6 = phys_derivs(dN_dξ₁_6, dN_dξ₂_6, Jinv11, Jinv12, Jinv21, Jinv22)
        wdetJq = QR3_w[q] * detJ
        btdb_shear!(ks, B_shear(N_ip3, dN_dx6, dN_dy6), gst, wdetJq)
    end

    # ── 9. Static condensation — eliminate midside w-DOFs (w4,w5,w6) ─────────
    # Partition ks (12×12):
    #   ks_aa = ks[1:9,  1:9 ]   vertex × vertex
    #   ks_ai = ks[1:9,  10:12]  vertex × midside
    #   ks_ii = ks[10:12,10:12]  midside × midside  (3×3, invertible)
    # ks_cond = ks_aa - ks_ai * inv(ks_ii) * ks_ia
    ks_ii     = MArray{Tuple{3,3},Float64,2,9}(undef)
    ks_ii_inv = MArray{Tuple{3,3},Float64,2,9}(undef)
    ks_cond   = MArray{Tuple{9,9},Float64,2,81}(undef)
    @inbounds for i in 1:3, j in 1:3
        ks_ii[i,j] = ks[9+i, 9+j]
    end
    inv3x3!(ks_ii_inv, ks_ii)
    @inbounds for r in 1:9
        for c in 1:9
            s = ks[r,c]
            for k in 1:3
                for l in 1:3
                    s -= ks[r, 9+k] * ks_ii_inv[k,l] * ks[9+l, c]
                end
            end
            ks_cond[r,c] = s
        end
    end

    # ── 10. Shear correction (Cs=0 → ke_bs = ks_cond + kb) ───────────────────
    # Matches CPU reference: Cs=0 disables blending, ke_bs = ks_cond + kb.
    ke_bs = MArray{Tuple{9,9},Float64,2,81}(undef)
    @inbounds for i in 1:9, j in 1:9
        ke_bs[i,j] = ks_cond[i,j] + kb[i,j]
    end

    # ── 11. Assemble ke_local (18×18, node-by-node ordering) ─────────────────
    ke_local = MArray{Tuple{18,18},Float64,2,324}(undef)
    fill!(ke_local, 0.0)
    @inbounds for r in 1:6, c in 1:6
        ke_local[INDUV18[r], INDUV18[c]] = km[r,c]
    end
    @inbounds for r in 1:9, c in 1:9
        ke_local[INDWT18[r], INDWT18[c]] = ke_bs[r,c]
    end
    # Drilling DOF penalty: stif = min of θx,θy diagonal entries / 100
    # ke_bs indices: [2]=θx1,[3]=θy1,[5]=θx2,[6]=θy2,[8]=θx3,[9]=θy3
    @inbounds stif = min(ke_bs[2,2],
                     min(ke_bs[3,3],
                     min(ke_bs[5,5],
                     min(ke_bs[6,6],
                     min(ke_bs[8,8], ke_bs[9,9]))))) / 100.0
    ke_local[6,  6 ] = stif
    ke_local[12, 12] = stif
    ke_local[18, 18] = stif

    # ── 12+13. Rotate ke_local → ke_all (fused, no ke_global temporary) ─────
    # write_ke_rotated_reordered! computes T*ke_local*T' and applies the
    # IND_FIELD permutation in a single pass, saving one 18×18 MArray
    # (324 Float64 = 648 registers) of local-memory pressure.
    write_ke_rotated_reordered!(ke_all, e, ke_local,
                                j1x, j1y, j1z,
                                j2x, j2y, j2z,
                                j3x, j3y, j3z)

    return nothing
end
