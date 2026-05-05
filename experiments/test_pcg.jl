# test_pcg.jl
#
# Verification tests for all three solver backends (A, B, C) in pcg_solver.jl.
#
# Purpose:
#   Confirm that all three solvers produce the same displacement u and that
#   Solver A's sub-kernels are individually correct. Run on small systems so
#   the ground-truth can be computed with Julia's direct solver (K \ f).
#
# ── Sub-kernel unit tests (Solver A) ─────────────────────────────────────────
#
#   spmv_kernel!:
#     Random sparse K (CSR), random x → compare GPU y = K*x against Julia result
#     Assert: norm(y_gpu - y_cpu) / norm(y_cpu) < 1e-12
#
#   dot_kernel!:
#     Random vectors a, b → compare GPU dot(a,b) against Julia dot(a,b)
#     Assert: abs(result_gpu - result_cpu) / abs(result_cpu) < 1e-12
#
#   axpy_kernel!:
#     Random alpha, x, y → compare GPU y += alpha*x against CPU result
#
#   axpby_kernel!:
#     Random alpha, beta, x, z → compare GPU y = alpha*x + beta*z against CPU
#
#   scale_kernel!:
#     Random alpha, x → compare GPU y = alpha*x against CPU
#
# ── Test Case 1 — Trivial diagonal system ────────────────────────────────────
#
#   K = Diagonal([1.0, 2.0, 4.0, ...])  (convert to CSR)
#   f = rand(n)
#   u_direct = K \ f                    (Julia reference)
#
#   Solver A: u_A = pcg_solve_A(K_csr..., f_d, tol=1e-12)
#   Solver B: u_B = pcg_solve_B(K_csr..., f_d, tol=1e-12)
#   Solver C: u_C = cusolver_solve(K_csr..., f_d)
#
#   Assert for each solver:
#     norm(u_X - u_direct) / norm(u_direct) < 1e-10
#   Assert cross-solver agreement:
#     norm(u_A - u_B) < 1e-10
#     norm(u_A - u_C) < 1e-10
#
# ── Test Case 2 — Small assembled system (2×2 mesh from test_small_mesh.jl) ──
#
#   Use K assembled by the GPU pipeline (element kernel + assembly kernel)
#   Apply Dirichlet BCs via dirichlet_kernel.jl (GPU)
#   Solve with all three backends and Julia's direct solver
#   Compare all four displacement vectors
#
#   Also verify: GPU Dirichlet result matches CPU apply_dirichlet! result
#     norm(K_gpu_bc - K_cpu_bc) / norm(K_cpu_bc) < 1e-12
#
# ── Test Case 3 — Convergence rate check (Solver A and B) ────────────────────
#
#   Use 2×2 mesh K (or slightly larger for meaningful iteration count)
#   Log residual norm every iteration → save as convergence data
#   Plot residual norm vs iteration count (log scale)
#   Expect roughly linear convergence for Jacobi-preconditioned PCG
#   Solver A and B should produce identical residual histories (same algorithm)
#
# ── Test Case 4 — Solver C fallback (CUSOLVER) ───────────────────────────────
#
#   Verify cusolver_solve calls csrlsvchol successfully for a well-conditioned K
#   Verify fallback to csrlsvqr if Cholesky reports non-SPD
#   (can test by artificially making a near-singular K)
#
# How to run:
#   julia --project=.. experiments/test_pcg.jl
#
