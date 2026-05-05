#!/usr/bin/env bash
# =============================================================================
# FEM Shell Solver — Euler SLURM job script
#
# Usage (single GPU):
#   1. Uncomment exactly ONE GPU block below (leave the rest double-commented).
#   2. sbatch experiments/run_fem_solver.sh
#
# Usage (all GPUs, benchmark sweep):
#   Run submit_fem_all_gpus.sh instead — it calls this script once per GPU type,
#   overriding the partition/gres via sbatch command-line flags.
#
# What this script does:
#   1. Prints environment info (node, GPU, Julia version, nvidia-smi).
#   2. Configures CUDA.jl to use the system toolkit (one-time, idempotent).
#   3. Runs test_single_element.jl  — validates GPU ke == CPU ke.
#   4. If a GPU_LABEL arg is present (set by submit_fem_all_gpus.sh), also
#      runs benchmark_fem_solver.jl and appends a row to benchmark_results_fem.csv.
# =============================================================================

#SBATCH --job-name=fem_solver
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00

# ── Uncomment exactly ONE GPU block ──────────────────────────────────────────
#
# ┌─ QUICK TEST (instruction partition, no reservation needed) ────────────────
#SBATCH -p instruction
#SBATCH --gres=gpu:rtx4000ada:1
# └────────────────────────────────────────────────────────────────────────────

# ┌─ RTX A4500  — Ampere — research partition ─────────────────────────────────
##SBATCH -p research
##SBATCH --gres=gpu:rtxa4500:1
# └────────────────────────────────────────────────────────────────────────────

# ┌─ RTX 2080 Ti — Turing — research partition ────────────────────────────────
##SBATCH -p research
##SBATCH --gres=gpu:rtx2080ti:1
# └────────────────────────────────────────────────────────────────────────────

# ┌─ A100  — Ampere — research partition ──────────────────────────────────────
##SBATCH -p research
##SBATCH --gres=gpu:a100:1
# └────────────────────────────────────────────────────────────────────────────

# ┌─ H100  — Hopper — research partition ──────────────────────────────────────
##SBATCH -p research
##SBATCH --gres=gpu:h100:1
# └────────────────────────────────────────────────────────────────────────────

# =============================================================================
set -euo pipefail

JULIA="$HOME/.juliaup/bin/julia"
PROJECT_DIR="$HOME/repo759/PROJECT"
RESULTS_CSV="$PROJECT_DIR/experiments/benchmark_results_fem.csv"

# GPU label: passed as $1 by submit_fem_all_gpus.sh, else fall back to job name
GPU_LABEL="${1:-${SLURM_JOB_NAME:-unknown}}"

cd "$PROJECT_DIR"

# ── Environment info ──────────────────────────────────────────────────────────
echo "============================================================"
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU label:  $GPU_LABEL"
echo "GPU SLURM:  ${SLURM_JOB_GPUS:-<not set>}"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "============================================================"

module load nvidia/cuda/13.0.0

"$JULIA" --version
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || nvidia-smi

echo ""
echo "── CPU info ─────────────────────────────────────────────────────────────"
lscpu | grep -E "Model name|Socket|Core\(s\) per socket|Thread|CPU MHz|NUMA"
echo "  logical CPUs available to job: ${SLURM_CPUS_ON_NODE:-$(nproc)}"

# ── Resolve all packages from Project.toml + Manifest.toml ───────────────────
# Pkg.instantiate() installs any packages that are registered but not yet
# downloaded on this compute node.  Requires setup_euler.jl to have been run
# once on the login node first (to register TriShellFiniteElement via Pkg.develop).
"$JULIA" --project=. -e 'import Pkg; Pkg.instantiate()'

# ── One-time CUDA.jl toolkit configuration ────────────────────────────────────
# Writes LocalPreferences.toml so CUDA.jl uses the system CUDA, not the JLL
# precompiled on the login node (which targets a different architecture).
"$JULIA" --project=. -e \
    'using CUDA; CUDA.set_runtime_version!(local_toolkit=true)' || true

echo ""
echo "── Step 1: Validate single-element ke (GPU == CPU) ─────────────────────"
"$JULIA" --project=. experiments/test_single_element.jl
echo ""

# ── Step 2: Full benchmark (only when called from submit_fem_all_gpus.sh) ────
# The benchmark script is passed the GPU label and CSV path so it can append
# a result row.  Skip gracefully if the script does not exist yet.
BENCH_SCRIPT="$PROJECT_DIR/experiments/benchmark_fem_solver.jl"
if [[ -f "$BENCH_SCRIPT" ]]; then
    echo "── Step 2: Benchmark fem solver ─────────────────────────────────────────"
    "$JULIA" --project=. "$BENCH_SCRIPT" "$GPU_LABEL" "$RESULTS_CSV"
    echo ""
    echo "Results appended → $RESULTS_CSV"
else
    echo "── Step 2: benchmark_fem_solver.jl not found — skipping."
    echo "   (create $BENCH_SCRIPT to enable full benchmarks)"
fi

echo ""
echo "============================================================"
echo "Done.  Output log: ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "============================================================"
