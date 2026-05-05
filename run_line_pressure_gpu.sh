#!/usr/bin/env bash
# =============================================================================
# SLURM job: run the GPU plate membrane line-pressure example
#
# Usage (from the FinalProject root on Euler):
#   sbatch run_line_pressure_gpu.sh
#
# Uncomment exactly ONE GPU block below before submitting.
# =============================================================================

#SBATCH --job-name=line_pressure_gpu
#SBATCH --output=line_pressure_%j.out
#SBATCH --error=line_pressure_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# ── Uncomment exactly ONE GPU block ──────────────────────────────────────────

# RTX 4000 Ada — instruction partition (default)
#SBATCH -p instruction
#SBATCH --gres=gpu:rtx4000ada:1

# RTX A4500 — research partition
##SBATCH -p research
##SBATCH --gres=gpu:rtxa4500:1

# RTX 2080 Ti — research partition
##SBATCH -p research
##SBATCH --gres=gpu:rtx2080ti:1

# A100 — research partition
##SBATCH -p research
##SBATCH --gres=gpu:a100:1

# H100 — research partition
##SBATCH -p research
##SBATCH --gres=gpu:h100:1

# =============================================================================

set -euo pipefail

JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/repo759/FinalProject}"
SCRIPT="$PROJECT_DIR/example/gpu_plate_membrane_line_pressure.jl"

cd "$PROJECT_DIR"

echo "============================================================"
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Project:    $PROJECT_DIR"
echo "Script:     $SCRIPT"
echo "GPU SLURM:  ${SLURM_JOB_GPUS:-<not set>}"
echo "============================================================"

module load nvidia/cuda/13.0.0

"$JULIA" --version
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || nvidia-smi

echo ""
echo "Instantiating Julia environment..."
"$JULIA" --project=. -e 'import Pkg; Pkg.instantiate()'

echo ""
echo "Configuring CUDA.jl to use the local CUDA toolkit..."
"$JULIA" --project=. -e 'using CUDA; CUDA.set_runtime_version!(local_toolkit=true)' || true

echo ""
echo "Running line-pressure GPU example..."
"$JULIA" --project=. "$SCRIPT"

echo ""
echo "Done."
