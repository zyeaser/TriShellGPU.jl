#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# GPU Benchmark Job Script
# Instructions:
#   1. Uncomment ONE of the GPU blocks below
#   2. Run: sbatch run_gpu_benchmark.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=julia_gpu_bench
#SBATCH --output=bench_%x_%j.out
#SBATCH --error=bench_%x_%j.err
#SBATCH --time=00:15:00

# ── Uncomment ONE GPU block ───────────────────────────────────────────────────

# RTX 4000 Ada — Ada Lovelace — instruction or research
#SBATCH -p instruction
#SBATCH --gres=gpu:rtx4000ada:1

# RTX A4500 — Ampere — research only
##SBATCH -p research
##SBATCH --gres=gpu:rtxa4500:1

# RTX 2080 Ti — Turing — research only
##SBATCH -p research
##SBATCH --gres=gpu:rtx2080ti:1

# A100 — Ampere — research only
##SBATCH -p research
##SBATCH --gres=gpu:a100:1

# H100 — Hopper — research only
##SBATCH -p research
##SBATCH --gres=gpu:h100:1

# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GPU_LABEL="${SLURM_JOB_NAME:-unknown}"
OUTPUT_CSV="$HOME/repo759/PROJECT/experiments/benchmark_results.csv"

JULIA="$HOME/.juliaup/bin/julia"
PROJECT_DIR="$HOME/repo759/PROJECT"
SCRIPT="$PROJECT_DIR/experiments/cuda_benchmark.jl"

cd "$PROJECT_DIR"

echo "=================================================="
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPU:      $SLURM_JOB_GPUS"
echo "=================================================="

module load nvidia/cuda/13.0.0

"$JULIA" --version
nvidia-smi

# Set local CUDA runtime (writes LocalPreferences.toml — one time setup)
"$JULIA" --project=. -e 'using CUDA; CUDA.set_runtime_version!(local_toolkit=true)' || true

# Run benchmark
"$JULIA" --project=. "$SCRIPT" "$SLURM_JOB_GPUS" "$OUTPUT_CSV"
