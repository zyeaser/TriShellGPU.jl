# setup_euler.jl
#
# One-time setup script — run this ONCE on Euler (login node) before submitting
# any SLURM jobs.  Safe to re-run; all operations are idempotent.
#
# Usage (from PROJECT/ directory):
#   ~/.juliaup/bin/julia --project=. experiments/setup_euler.jl
#
# What it does:
#   1. Adds all missing registered packages (StaticArrays, etc.)
#   2. Registers the local TriShellFiniteElement package via Pkg.develop
#   3. Instantiates / precompiles everything
#   4. Configures CUDA.jl to use the system toolkit on Euler

import Pkg

PROJECT_DIR = joinpath(@__DIR__, "..")
cd(PROJECT_DIR)

println("=" ^ 60)
println("Euler one-time project setup")
println("Project: ", PROJECT_DIR)
println("Julia:   ", VERSION)
println("=" ^ 60)
println()

# ── Step 1: Add missing registered packages ───────────────────────────────────
println("── Step 1: Adding registered packages...")

registered = ["StaticArrays", "Ferrite", "CUDA", "BenchmarkTools", "Tensors"]
for pkg in registered
    try
        Pkg.add(pkg)
        println("  ✓  $pkg")
    catch e
        println("  (already present or skipped: $pkg)")
    end
end
println()

# ── Step 2: Develop local TriShellFiniteElement package ───────────────────────
# Pkg.develop registers a local path package into the project so that
# `using TriShellFiniteElement` works in any script without Pkg.activate tricks.
println("── Step 2: Registering local TriShellFiniteElement package...")
trishell_path = joinpath(PROJECT_DIR, "resources", "TriShellFiniteElement.jl")
if isdir(trishell_path)
    Pkg.develop(path=trishell_path)
    println("  ✓  TriShellFiniteElement  →  $trishell_path")
else
    println("  ✗  Not found at: $trishell_path")
    println("     Check that resources/TriShellFiniteElement.jl/ exists.")
end
println()

# ── Step 3: Instantiate (resolve + download all deps) ─────────────────────────
println("── Step 3: Instantiating project (resolving all dependencies)...")
Pkg.instantiate()
println("  ✓  Done")
println()

# ── Step 4: Precompile ────────────────────────────────────────────────────────
println("── Step 4: Precompiling packages (this may take a few minutes)...")
Pkg.precompile()
println("  ✓  Done")
println()

# ── Step 5: Configure CUDA.jl to use Euler's system toolkit ──────────────────
println("── Step 5: Configuring CUDA.jl for system toolkit...")
try
    using CUDA
    CUDA.set_runtime_version!(local_toolkit=true)
    println("  ✓  CUDA.jl set to local_toolkit=true")
    println("     (restart Julia / new job for this to take effect)")
catch e
    println("  ! CUDA config skipped: $e")
end
println()

println("=" ^ 60)
println("Setup complete.  You can now submit SLURM jobs:")
println("  sbatch experiments/run_fem_solver.sh")
println("=" ^ 60)
