# test_small_mesh.jl
#
# Verification test: GPU global stiffness Ke == CPU global stiffness Ke
# for a small 2×2 mesh (8 triangles, ~9 nodes).
#
# Purpose:
#   After test_single_element.jl passes, verify that assembly_kernel! correctly
#   scatters all element ke's into the global CSR matrix K.
#   A 2×2 mesh is small enough to inspect manually but exercises the atomic-add
#   scatter logic across shared DOFs.
#
# Test setup:
#   - Generate a 2×2 plate mesh using Ferrite.jl (or define manually)
#   - Assemble K_cpu using the CPU reference code (direct sparse assembly)
#   - Assemble K_gpu using gpu_solve pipeline: element kernel + assembly kernel
#   - Download K_gpu values back to CPU and compare with K_cpu
#
# What to check:
#   - K_cpu and K_gpu have identical sparsity pattern
#   - norm(K_cpu - K_gpu) / norm(K_cpu) < tolerance (e.g. 1e-10)
#   - K is symmetric: norm(K - K') < 1e-12
#   - K is positive semi-definite (before BCs): all eigenvalues ≥ 0
#
# How to run:
#   julia --project=.. experiments/test_small_mesh.jl
#
