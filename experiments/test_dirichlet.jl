# test_dirichlet.jl
#
# Verification test: GPU Dirichlet BC application matches CPU reference result.
#
# Purpose:
#   Confirm that dirichlet_row_kernel! + dirichlet_col_kernel! (from dirichlet_kernel.jl)
#   produce the same modified K and f as the CPU apply_dirichlet! function.
#   This must pass before any solver test is trusted, because all three solver
#   backends receive K and f after GPU BC application.
#
# Setup:
#   - Assemble K_gpu and f_gpu for the 2×2 mesh (from test_small_mesh.jl)
#   - Download K_gpu → K_cpu_before (pre-BC reference)
#   - Apply BCs on CPU: apply_dirichlet!(K_cpu, rowptr, colind, f_cpu, fixed_dofs)
#   - Apply BCs on GPU: dirichlet_row_kernel! + dirichlet_col_kernel!
#   - Download K_gpu_after, f_gpu_after
#
# What to check:
#
#   1. Row zeroing:
#        For each fixed DOF g, all entries in row g of K_gpu_after are zero
#        except K[g,g] = 1.0
#        Assert: norm(K_gpu_after[g, :] - e_g) == 0
#
#   2. Column zeroing:
#        For each fixed DOF g, all entries in column g of K_gpu_after are zero
#        except K[g,g] = 1.0
#        Assert: norm(K_gpu_after[:, g] - e_g) == 0
#
#   3. Symmetry preserved:
#        norm(K_gpu_after - K_gpu_after') < 1e-12
#
#   4. Match CPU result:
#        norm(K_gpu_after - K_cpu_after) / norm(K_cpu_after) < 1e-12
#        norm(f_gpu_after - f_cpu_after) < 1e-12
#
#   5. diag_idx correctness:
#        For each DOF g, K_values[diag_idx[g]] corresponds to K[g,g]
#        Verify by downloading K_values and checking against dense K diagonal
#
#   6. col_ptr / col_vals correctness:
#        For a small known mesh, manually verify that col_vals contains the
#        correct flat CSR indices for each fixed DOF's column
#
# How to run:
#   julia --project=.. experiments/test_dirichlet.jl
#
# Dependencies:
#   Requires test_small_mesh.jl to have passed (same mesh and assembly used here).
#