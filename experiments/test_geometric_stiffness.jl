# test_geometric_stiffness.jl
#
# Verification test: GPU geometric stiffness Kg == CPU geometric stiffness Kg.
#
# Purpose:
#   Confirm that geometric_stiffness_kernel! produces the same (n_elem, 18, 18)
#   kg_all and that assembly_kernel! (reused) correctly assembles the global Kg,
#   matching the CPU reference assemble_global_Kg! (with the bug fixed).
#
# Background:
#   The CPU reference assemble_global_Kg! has a bug at lines 605-607:
#     σxx_element, σyy_element, τxy_element are referenced but never assigned.
#   The GPU kernel fixes this by using σXX[i], σYY[i], τXY[i] directly.
#   This test also implicitly verifies that the fix is correct.
#
# Test setup:
#   - Use the 2×2 mesh from test_small_mesh.jl
#   - Run the full static solve (GPU) to get displacement u
#   - Compute element stresses σ from u on CPU:
#       σ_elem = compute_element_stresses(u, mesh_params, material)
#       → (n_elem, 3) array of [σxx, σyy, τxy] per element in global coords
#   - Assemble Kg_cpu using the FIXED CPU reference (with σXX/σYY/τXY)
#   - Assemble Kg_gpu using geometric_stiffness_kernel! + assembly_kernel!
#   - Download Kg_gpu and compare
#
# What to check:
#
#   1. Per-element geometric stiffness:
#        For a single element with known coords and stresses,
#        compute kg_cpu (CPU reference) and kg_gpu (GPU kernel output from kg_all)
#        Assert: norm(kg_gpu - kg_cpu) / norm(kg_cpu) < 1e-10
#
#   2. Global Kg assembly:
#        norm(Kg_gpu - Kg_cpu) / norm(Kg_cpu) < 1e-10
#
#   3. Kg symmetry:
#        norm(Kg_gpu - Kg_gpu') < 1e-12
#
#   4. Stress transformation correctness:
#        Verify that T2 rotation (lines 548-551 in CPU reference) is correctly
#        reproduced in the GPU kernel for a non-trivial element orientation
#        (element not aligned with global X axis)
#
#   5. CSR sparsity reuse:
#        Verify that the same rowptr/colind/scatter_idx used for K assembly
#        work correctly for Kg assembly (same sparsity pattern, different values)
#
#   6. Bug fix verification:
#        Run CPU reference with the BUG (σxx_element undefined) → confirm error/wrong result
#        Run CPU reference with the FIX (σXX_element) → confirm matches GPU
#
# How to run:
#   julia --project=.. experiments/test_geometric_stiffness.jl
#
# Dependencies:
#   Requires test_small_mesh.jl and test_pcg.jl to have passed.
#   Stress computation function must be implemented in gpu_solve.jl (Step A of buckling).
#