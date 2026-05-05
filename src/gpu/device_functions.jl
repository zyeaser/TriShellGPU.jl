# device_functions.jl — Step 1
#
# Pure math helpers called inside GPU kernels (no kernel launches here).
# Every function is @inline so the compiler folds them into the calling kernel.
# No heap allocation: all state lives in registers / GPU local memory.
#
# Conventions
#   - NTuple matrices are stored ROW-MAJOR.
#     A 3×6 matrix M has M[r,c] = M_tuple[(r-1)*6 + c].
#   - 3-D vectors are passed as (x, y, z) scalar triples.
#   - 2-D vectors are passed as (x, y) scalar pairs.
#   - T3 (3×3 rotation) is passed as its three COLUMN vectors
#     (j1x,j1y,j1z, j2x,j2y,j2z, j3x,j3y,j3z).
#
# All functions match the CPU reference in
#   TriShellFiniteElement.jl/src/TriShellFiniteElement.jl
# exactly, including DOF ordering, constitutive scaling, and index conventions.

# ── Quadrature rules ──────────────────────────────────────────────────────────
#
# qr1 — 1-point centroid rule (used for membrane and bending)
const QR1_ξ₁ = (1.0/3.0,)
const QR1_ξ₂ = (1.0/3.0,)
const QR1_w  = (1.0/2.0,)

# qr3 — Ferrite QuadratureRule{RefTriangle}(3)  (degree-3, 4 points)
#
#   Matches exactly what Ferrite produces in the REPL:
#     qr3.points  = [[1/3,1/3], [0.2,0.2], [0.2,0.6], [0.6,0.2]]
#     qr3.weights = [-0.28125, 0.260416…, 0.260416…, 0.260416…]
#
#   Point 1 sits at the centroid with a negative weight; this is a standard
#   property of the Dunavant degree-3 rule — the weights sum to 0.5 (triangle
#   reference area) and the rule integrates cubics exactly.
#
#   *** Replaces the old 3-point midpoint rule: ***
#     OLD: QR3_ξ₁=(1/6,2/3,1/6), QR3_ξ₂=(1/6,1/6,2/3), QR3_w=(1/6,1/6,1/6)
#          — that rule only integrates quadratics exactly; it did NOT match
#            Ferrite's qr3 and produced different shear stiffness than the CPU.
#     NEW: 4 points, matches Ferrite QuadratureRule{RefTriangle}(3) exactly.
const QR3_ξ₁ = (0.33333333333333, 0.2, 0.2, 0.6)
const QR3_ξ₂ = (0.33333333333333, 0.2, 0.6, 0.2)
const QR3_w  = (-0.28125, 0.260416666666665, 0.260416666666665, 0.260416666666665)

# ── DOF reordering for Ferrite assembly ───────────────────────────────────────
#
# After ke_global is assembled in node-by-node order
#   Node k: [uk, vk, wk, θxk, θyk, θzk]  →  indices 6(k-1)+1 .. 6k
# this permutation converts to Ferrite field-by-field order
#   :u field (u1,v1,w1, u2,v2,w2, u3,v3,w3) at positions 1-9
#   :θ field (θx1,θy1,θz1, θx2,θy2,θz2, θx3,θy3,θz3) at positions 10-18
#
# ke_ferrite[i,j] = ke_nodewise[ IND_FIELD[i], IND_FIELD[j] ]
#
# Matches CPU reference: assemble_global_Ke! line 429.
const IND_FIELD = (1, 2, 3, 7, 8, 9, 13, 14, 15, 4, 5, 6, 10, 11, 12, 16, 17, 18)

# ── Shape functions ───────────────────────────────────────────────────────────

# IP3 — 3-node linear triangle.
# Used for: membrane (u,v), bending (w,θx,θy), and element geometry.
#
#   N1 = 1 - ξ₁ - ξ₂
#   N2 = ξ₁
#   N3 = ξ₂
#
# Returns: N (3,), dN_dξ₁ (3,), dN_dξ₂ (3,)
# Derivatives are constants — they do not depend on (ξ₁, ξ₂).
@inline function shape_functions_ip3(ξ₁::T, ξ₂::T) where {T}
    N      = (one(T) - ξ₁ - ξ₂,  ξ₁,    ξ₂)
    dN_dξ₁ = (-one(T),  one(T), zero(T))
    dN_dξ₂ = (-one(T), zero(T),  one(T))
    return N, dN_dξ₁, dN_dξ₂
end

# IP6 — 6-node mixed triangle (vertex nodes use linear basis, midside use bubble).
# Used for: shear w-field interpolation (MITC approach).
#
# Node ordering (matches Ferrite IP6):
#   1,2,3 = vertices  →  N1=1-ξ₁-ξ₂, N2=ξ₁, N3=ξ₂  (same as IP3)
#   4,5,6 = midside nodes on edges 1-2, 1-3, 2-3
#       N4 = 4 ξ₁(1-ξ₁-ξ₂)
#       N5 = 4 ξ₁ ξ₂
#       N6 = 4 ξ₂(1-ξ₁-ξ₂)
#
# Returns: N (6,), dN_dξ₁ (6,), dN_dξ₂ (6,)
@inline function shape_functions_ip6(ξ₁::T, ξ₂::T) where {T}
    N = (
        one(T) - ξ₁ - ξ₂,
        ξ₁,
        ξ₂,
        T(4) * ξ₁ * (one(T) - ξ₁ - ξ₂),
        T(4) * ξ₁ * ξ₂,
        T(4) * ξ₂ * (one(T) - ξ₁ - ξ₂)
    )
    dN_dξ₁ = (
        -one(T),
         one(T),
        zero(T),
        T(4) * (one(T) - T(2)*ξ₁ - ξ₂),
        T(4) * ξ₂,
       -T(4) * ξ₂
    )
    dN_dξ₂ = (
        -one(T),
        zero(T),
         one(T),
        -T(4) * ξ₁,
         T(4) * ξ₁,
        T(4) * (one(T) - ξ₁ - T(2)*ξ₂)
    )
    return N, dN_dξ₁, dN_dξ₂
end

# ── Jacobian ──────────────────────────────────────────────────────────────────
#
# Compute the inverse 2×2 Jacobian and its determinant from the 3-node linear
# geometry (IP3).  For a linear triangle J is CONSTANT across the element, so
# this is called once per element, not once per quadrature point.
#
# (x1,y1), (x2,y2), (x3,y3) — 2-D local coordinates of the three vertices
#   (output of project_to_local, taking only the first two components)
#
# IP3 natural derivatives are constants:
#   dN/dξ₁ = (−1, 1, 0)
#   dN/dξ₂ = (−1, 0, 1)
#
# J = [ J11  J12 ] = [ −x1+x2   −y1+y2 ]
#     [ J21  J22 ]   [ −x1+x3   −y1+y3 ]
#
# Returns: (Jinv11, Jinv12, Jinv21, Jinv22, detJ)
@inline function compute_jacobian(
        x1::T, y1::T,
        x2::T, y2::T,
        x3::T, y3::T) where {T}
    J11 = x2 - x1
    J12 = y2 - y1
    J21 = x3 - x1
    J22 = y3 - y1
    detJ   = J11 * J22 - J12 * J21
    inv_d  = one(T) / detJ
    Jinv11 =  J22 * inv_d
    Jinv12 = -J12 * inv_d
    Jinv21 = -J21 * inv_d
    Jinv22 =  J11 * inv_d
    return Jinv11, Jinv12, Jinv21, Jinv22, detJ
end

# Physical (x̄,ȳ) derivatives from natural-coordinate derivatives + J⁻¹.
# Works for both IP3 (N=3) and IP6 (N=6) — the Jacobian is always from IP3
# geometry because the element mapping is linear.
#
# dN_dx[i] = Jinv11 * dN_dξ₁[i] + Jinv12 * dN_dξ₂[i]
# dN_dy[i] = Jinv21 * dN_dξ₁[i] + Jinv22 * dN_dξ₂[i]
#
# Returns: dN_dx (N,), dN_dy (N,)
@inline function phys_derivs(
        dN_dξ₁::NTuple{N,T},
        dN_dξ₂::NTuple{N,T},
        Jinv11::T, Jinv12::T,
        Jinv21::T, Jinv22::T) where {N,T}
    dN_dx = ntuple(i -> Jinv11 * dN_dξ₁[i] + Jinv12 * dN_dξ₂[i], Val(N))
    dN_dy = ntuple(i -> Jinv21 * dN_dξ₁[i] + Jinv22 * dN_dξ₂[i], Val(N))
    return dN_dx, dN_dy
end

# ── Constitutive matrices ─────────────────────────────────────────────────────
#
# All returned as NTuple in row-major order.
# Matches CPU reference: calculate_membrane/bending/shear_constitutive_matrix().

# Membrane: 3×3 Dm  (NTuple{9})
#   Dm = (E·t / (1−ν²)) * [ 1   ν    0        ]
#                           [ ν   1    0        ]
#                           [ 0   0   (1−ν)/2   ]
# Note: scaled by t here — do NOT multiply by t again in the quadrature loop.
@inline function constitutive_membrane(E::T, ν::T, t::T) where {T}
    c  = E * t / (one(T) - ν * ν)
    cg = c * (one(T) - ν) / T(2)
    z  = zero(T)
    return (c,   c*ν,  z,
            c*ν, c,    z,
            z,   z,    cg)
end

# Bending: 3×3 Db  (NTuple{9})
#   Db = (E·t³ / (12(1−ν²))) * [ 1   ν    0        ]
#                                [ ν   1    0        ]
#                                [ 0   0   (1−ν)/2   ]
@inline function constitutive_bending(E::T, ν::T, t::T) where {T}
    c  = E * t^3 / (T(12) * (one(T) - ν * ν))
    cg = c * (one(T) - ν) / T(2)
    z  = zero(T)
    return (c,   c*ν,  z,
            c*ν, c,    z,
            z,   z,    cg)
end

# Shear: Ds = (5/6)·G·t · I₂, returned as the single diagonal value.
#   G = E / (2(1+ν))
#   gst = (5/6) · G · t
# Both diagonal entries are equal; off-diagonal are zero.
@inline function constitutive_shear_diag(E::T, ν::T, t::T) where {T}
    G = E / (T(2) * (one(T) + ν))
    return T(5) / T(6) * G * t          # gst
end

# ── B matrices ────────────────────────────────────────────────────────────────

# B_membrane — 3×6, DOFs: [u1, v1, u2, v2, u3, v3]
#
#   B = [ dN₁/dx̄   0       dN₂/dx̄   0       dN₃/dx̄   0      ]   row 1: ε_xx
#       [ 0        dN₁/dȳ  0        dN₂/dȳ  0        dN₃/dȳ ]   row 2: ε_yy
#       [ dN₁/dȳ  dN₁/dx̄  dN₂/dȳ  dN₂/dx̄  dN₃/dȳ  dN₃/dx̄ ]   row 3: γ_xy
#
# Matches CPU reference: calculate_element_membrane_stiffness_matrix, B_node loop.
# Returns NTuple{18,T} (row-major).
@inline function B_membrane(dN_dx::NTuple{3,T}, dN_dy::NTuple{3,T}) where {T}
    z = zero(T)
    return (
        dN_dx[1], z,        dN_dx[2], z,        dN_dx[3], z,
        z,        dN_dy[1], z,        dN_dy[2], z,        dN_dy[3],
        dN_dy[1], dN_dx[1], dN_dy[2], dN_dx[2], dN_dy[3], dN_dx[3]
    )
end

# B_bending — 3×9, DOFs: [w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3]
#
#   B = [ 0   0        dN₁/dx̄   0   0        dN₂/dx̄   0   0        dN₃/dx̄  ]   κ_xx
#       [ 0  −dN₁/dȳ   0        0  −dN₂/dȳ   0        0  −dN₃/dȳ   0       ]   κ_yy
#       [ 0  −dN₁/dx̄   dN₁/dȳ  0  −dN₂/dx̄   dN₂/dȳ  0  −dN₃/dx̄   dN₃/dȳ ]   κ_xy
#
# Matches CPU reference: calculate_element_bending_stiffness_matrix, B_node.
# Returns NTuple{27,T} (row-major).
@inline function B_bending(dN_dx::NTuple{3,T}, dN_dy::NTuple{3,T}) where {T}
    z = zero(T)
    return (
        z, z,          dN_dx[1],  z, z,          dN_dx[2],  z, z,          dN_dx[3],
        z, -dN_dy[1],  z,         z, -dN_dy[2],  z,         z, -dN_dy[3],  z,
        z, -dN_dx[1],  dN_dy[1],  z, -dN_dx[2],  dN_dy[2],  z, -dN_dx[3],  dN_dy[3]
    )
end

# B_shear — 2×12, DOFs: [w1,θx1,θy1, w2,θx2,θy2, w3,θx3,θy3, w4, w5, w6]
#
# Columns 1-9  : vertex node k (k=1,2,3) — each contributes a 2×3 block:
#   [ dNₖ/dx̄   0    Nₖ  ]   row 1: γ_xz
#   [ dNₖ/dȳ  −Nₖ   0   ]   row 2: γ_yz
#   (N from IP3 vertex shape values; dN from IP6 physical derivatives)
#
# Columns 10-12: midside nodes 4,5,6 — each contributes a 2×1 column:
#   [ dNₖ/dx̄ ]
#   [ dNₖ/dȳ ]
#   (θ-DOFs do not exist for midside nodes; only w-DOF)
#
# This is the 2×12 matrix obtained after extracting active DOFs [1:10;13;16]
# from the full 2×18 B assembled in the CPU reference's
# calculate_element_shear_stiffness_matrix.
#
# N_ip3     : IP3 (= IP6 node 1-3) shape values at the quadrature point
# dN_dx_ip6 : IP6 physical x-derivatives at the quadrature point (length 6)
# dN_dy_ip6 : IP6 physical y-derivatives at the quadrature point (length 6)
# Returns NTuple{24,T} (row-major).
@inline function B_shear(
        N_ip3    ::NTuple{3,T},
        dN_dx_ip6::NTuple{6,T},
        dN_dy_ip6::NTuple{6,T}) where {T}
    z = zero(T)
    return (
        # row 1 — γ_xz
        dN_dx_ip6[1], z,           N_ip3[1],
        dN_dx_ip6[2], z,           N_ip3[2],
        dN_dx_ip6[3], z,           N_ip3[3],
        dN_dx_ip6[4], dN_dx_ip6[5], dN_dx_ip6[6],
        # row 2 — γ_yz
        dN_dy_ip6[1], -N_ip3[1],  z,
        dN_dy_ip6[2], -N_ip3[2],  z,
        dN_dy_ip6[3], -N_ip3[3],  z,
        dN_dy_ip6[4], dN_dy_ip6[5], dN_dy_ip6[6]
    )
end

# ── B'DB product helpers ──────────────────────────────────────────────────────
#
# These are called inside the quadrature loop of element_stiffness_kernel!.
# Each function accumulates  ke_block[r,c] += Σ_{i,j} B[i,r]·D[i,j]·B[j,c]·wdetJ
# where B is stored row-major in a NTuple.
#
# The accumulator arrays (km, kb, ks) are MArrays living in GPU local memory;
# they are declared and zeroed in the kernel before the quadrature loop.

# btdb_membrane!
#   B    : NTuple{18} — 3×6 B_membrane (row-major)
#   D    : NTuple{9}  — 3×3 Dm (row-major, symmetric)
#   km   : 6×6 mutable accumulator (e.g. MArray{Tuple{6,6},Float64})
#   wdetJ : quadrature weight × |J|
@inline function btdb_membrane!(km, B::NTuple{18,T}, D::NTuple{9,T}, wdetJ::T) where {T}
    @inbounds for r in 1:6
        for c in 1:6
            s = zero(T)
            for i in 1:3
                Bir = B[(i-1)*6 + r]
                for j in 1:3
                    s += Bir * D[(i-1)*3 + j] * B[(j-1)*6 + c]
                end
            end
            km[r, c] += s * wdetJ
        end
    end
end

# btdb_bending!
#   B    : NTuple{27} — 3×9 B_bending (row-major)
#   D    : NTuple{9}  — 3×3 Db (row-major, symmetric)
#   kb   : 9×9 mutable accumulator
@inline function btdb_bending!(kb, B::NTuple{27,T}, D::NTuple{9,T}, wdetJ::T) where {T}
    @inbounds for r in 1:9
        for c in 1:9
            s = zero(T)
            for i in 1:3
                Bir = B[(i-1)*9 + r]
                for j in 1:3
                    s += Bir * D[(i-1)*3 + j] * B[(j-1)*9 + c]
                end
            end
            kb[r, c] += s * wdetJ
        end
    end
end

# btdb_shear!
#   B_s  : NTuple{24} — 2×12 B_shear (row-major)
#   gst  : scalar diagonal of 2×2 Ds = gst · I  (from constitutive_shear_diag)
#   ks   : 12×12 mutable accumulator
#
# Because Ds is diagonal, B'·Ds·B simplifies to:
#   ks[r,c] = gst · ( B_s[1,r]·B_s[1,c] + B_s[2,r]·B_s[2,c] )
# where B_s[1,r] = B_s[r] and B_s[2,r] = B_s[12+r]  (row-major, 12 cols).
@inline function btdb_shear!(ks, B_s::NTuple{24,T}, gst::T, wdetJ::T) where {T}
    coeff = gst * wdetJ
    @inbounds for r in 1:12
        B1r = B_s[r]
        B2r = B_s[12 + r]
        for c in 1:12
            ks[r, c] += coeff * (B1r * B_s[c] + B2r * B_s[12 + c])
        end
    end
end

# ── Coordinate rotation ───────────────────────────────────────────────────────

# local_to_global_rotation
#   Inputs : three global 3-D node coordinates P1, P2, P3
#   Builds element local frame:
#       j1 = (P2−P1) / ‖P2−P1‖              (local x̄, along edge 1→2)
#       j3 = (P2−P1) × (P3−P1) / ‖·‖        (element normal)
#       j2 = j3 × j1                          (local ȳ, completes right-hand frame)
#       T  = [j1 | j2 | j3]                   (3×3, columns are local basis vectors)
#   Matches CPU reference: calculation_rotation_matrix().
#
# Returns column vectors j1, j2, j3 each as (x,y,z) scalars (9 values total).
@inline function local_to_global_rotation(
        P1x::T, P1y::T, P1z::T,
        P2x::T, P2y::T, P2z::T,
        P3x::T, P3y::T, P3z::T) where {T}
    # v12 = P2 - P1
    v12x = P2x - P1x;  v12y = P2y - P1y;  v12z = P2z - P1z
    len12 = sqrt(v12x*v12x + v12y*v12y + v12z*v12z)
    j1x = v12x / len12;  j1y = v12y / len12;  j1z = v12z / len12

    # v13 = P3 - P1
    v13x = P3x - P1x;  v13y = P3y - P1y;  v13z = P3z - P1z

    # j3 = cross(v12, v13) / ‖·‖  (element normal)
    nx = v12y*v13z - v12z*v13y
    ny = v12z*v13x - v12x*v13z
    nz = v12x*v13y - v12y*v13x
    len_n = sqrt(nx*nx + ny*ny + nz*nz)
    j3x = nx / len_n;  j3y = ny / len_n;  j3z = nz / len_n

    # j2 = cross(j3, j1)
    j2x = j3y*j1z - j3z*j1y
    j2y = j3z*j1x - j3x*j1z
    j2z = j3x*j1y - j3y*j1x

    return j1x, j1y, j1z,
           j2x, j2y, j2z,
           j3x, j3y, j3z
end

# project_to_local
#   Project three global 3-D node coordinates into the element's 2-D local plane.
#   x_local_i = T' * (Pi − P1),  keeping only the first two components.
#   P1 maps to origin.
#   Matches CPU reference: global_nodal_coords_to_planar_coords().
#
# T columns : (j1x,j1y,j1z, j2x,j2y,j2z) — only j1 and j2 needed here
# Returns (x1,y1, x2,y2, x3,y3) in local 2-D coordinates.
@inline function project_to_local(
        P1x::T, P1y::T, P1z::T,
        P2x::T, P2y::T, P2z::T,
        P3x::T, P3y::T, P3z::T,
        j1x::T, j1y::T, j1z::T,
        j2x::T, j2y::T, j2z::T) where {T}
    # P1 → origin
    lx1 = zero(T);  ly1 = zero(T)
    # P2 − P1
    d2x = P2x - P1x;  d2y = P2y - P1y;  d2z = P2z - P1z
    lx2 = j1x*d2x + j1y*d2y + j1z*d2z
    ly2 = j2x*d2x + j2y*d2y + j2z*d2z
    # P3 − P1
    d3x = P3x - P1x;  d3y = P3y - P1y;  d3z = P3z - P1z
    lx3 = j1x*d3x + j1y*d3y + j1z*d3z
    ly3 = j2x*d3x + j2y*d3y + j2z*d3z
    return lx1, ly1, lx2, ly2, lx3, ly3
end

# rotate_block3x3
#   Compute  C = T3 * A * T3'  for a 3×3 matrix A.
#   Used to rotate each 3×3 block of ke_local into global coordinates.
#
# A        : NTuple{9,T}  row-major  (A[i,j] = A[(i-1)*3 + j])
# j1,j2,j3 : columns of T3  (each a 3-vector)
#
# C[i,k] = Σ_{a,b}  T3[i,a] · A[a,b] · T3[k,b]
#
# Returns NTuple{9,T} row-major.
@inline function rotate_block3x3(
        A  ::NTuple{9,T},
        j1x::T, j1y::T, j1z::T,
        j2x::T, j2y::T, j2z::T,
        j3x::T, j3y::T, j3z::T) where {T}

    # T3[row i, col a]:  col 1 = j1, col 2 = j2, col 3 = j3
    @inline t3(i,a) = ifelse(a==1, ifelse(i==1, j1x, ifelse(i==2, j1y, j1z)),
                      ifelse(a==2, ifelse(i==1, j2x, ifelse(i==2, j2y, j2z)),
                                   ifelse(i==1, j3x, ifelse(i==2, j3y, j3z))))
    @inline Aab(a,b) = A[(a-1)*3 + b]

    ntuple(Val(9)) do k_flat
        i  = (k_flat - 1) ÷ 3 + 1
        kk = (k_flat - 1) % 3 + 1
        s  = zero(T)
        for a in 1:3
            T3ia = t3(i, a)
            for b in 1:3
                s += T3ia * Aab(a, b) * t3(kk, b)
            end
        end
        s
    end
end

# rotate_ke18!
#   Apply the 18×18 block-diagonal rotation  ke_global = Te · ke_local · Te'
#   where  Te = blkdiag(T3, T3, T3, T3, T3, T3)  (six 3×3 blocks).
#
# ke_local and ke_out are 18×18 mutable arrays (MArray in the kernel).
# The six 3×3 groups correspond to DOF triplets:
#   group 1: rows/cols 1-3   (node 1 u,v,w)
#   group 2: rows/cols 4-6   (node 1 θx,θy,θz)
#   group 3: rows/cols 7-9   (node 2 u,v,w)
#   group 4: rows/cols 10-12 (node 2 θx,θy,θz)
#   group 5: rows/cols 13-15 (node 3 u,v,w)
#   group 6: rows/cols 16-18 (node 3 θx,θy,θz)
#
# Matches CPU reference: rotation_matrix_for_element_stiffness_drilling()
# followed by  ke_global = Te * ke_local * Te'.
@inline function rotate_ke18!(ke_out, ke_local,
        j1x::T, j1y::T, j1z::T,
        j2x::T, j2y::T, j2z::T,
        j3x::T, j3y::T, j3z::T) where {T}

    @inline t3(i,a) = ifelse(a==1, ifelse(i==1, j1x, ifelse(i==2, j1y, j1z)),
                      ifelse(a==2, ifelse(i==1, j2x, ifelse(i==2, j2y, j2z)),
                                   ifelse(i==1, j3x, ifelse(i==2, j3y, j3z))))

    # ke_out[3(I-1)+i, 3(J-1)+k] = Σ_{a,b} T3[i,a] · ke_local[3(I-1)+a, 3(J-1)+b] · T3[k,b]
    @inbounds for I in 1:6
        ri = 3*(I-1)
        for J in 1:6
            rj = 3*(J-1)
            for i in 1:3
                for kk in 1:3
                    s = zero(T)
                    for a in 1:3
                        T3ia = t3(i, a)
                        for b in 1:3
                            s += T3ia * ke_local[ri+a, rj+b] * t3(kk, b)
                        end
                    end
                    ke_out[ri+i, rj+kk] = s
                end
            end
        end
    end
end

# write_ke_rotated_reordered!
#   Fused rotation + IND_FIELD reordering written directly to ke_all.
#   Eliminates the ke_global MArray (18×18, 648 registers) by computing each
#   output entry on-the-fly and streaming it straight to global memory.
#
# Equivalence:
#   ke_global = Te * ke_local * Te'           (rotate_ke18!)
#   ke_all[e,fi,fj] = ke_global[IND_FIELD[fi], IND_FIELD[fj]]
#
# combined into a single loop with no ke_global temporary.
#
# For each Ferrite output index pair (fi, fj):
#   p = IND_FIELD[fi]   — nodewise row index  (1-18)
#   q = IND_FIELD[fj]   — nodewise col index  (1-18)
#   I = (p-1)÷3 + 1,  i_loc = (p-1)%3 + 1   — block and within-block for row
#   J = (q-1)÷3 + 1,  k_loc = (q-1)%3 + 1   — block and within-block for col
#   ke_all[e,fi,fj] = Σ_{a,b} T3[i_loc,a] * ke_local[3(I-1)+a, 3(J-1)+b] * T3[k_loc,b]
#
@inline function write_ke_rotated_reordered!(
        ke_all   ::CuDeviceArray{Float64,3},
        e        ::Int32,
        ke_local,
        j1x::T, j1y::T, j1z::T,
        j2x::T, j2y::T, j2z::T,
        j3x::T, j3y::T, j3z::T) where {T}

    @inline t3(i,a) = ifelse(a==1, ifelse(i==1, j1x, ifelse(i==2, j1y, j1z)),
                      ifelse(a==2, ifelse(i==1, j2x, ifelse(i==2, j2y, j2z)),
                                   ifelse(i==1, j3x, ifelse(i==2, j3y, j3z))))

    @inbounds for fi in 1:18
        p   = IND_FIELD[fi]
        I   = (p - 1) ÷ 3 + 1      # block index 1-6
        i_loc = (p - 1) % 3 + 1    # position within block 1-3
        ri  = 3 * (I - 1)

        for fj in 1:18
            q   = IND_FIELD[fj]
            J   = (q - 1) ÷ 3 + 1
            k_loc = (q - 1) % 3 + 1
            rj  = 3 * (J - 1)

            s = zero(T)
            for a in 1:3
                T3ia = t3(i_loc, a)
                for b in 1:3
                    s += T3ia * ke_local[ri+a, rj+b] * t3(k_loc, b)
                end
            end
            ke_all[e, fi, fj] = s
        end
    end
end
