#!/usr/bin/env bash
#SBATCH --job-name=julia_cuda_test
#SBATCH --output=cuda_test.out
#SBATCH --error=cuda_test.err
#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

set -euo pipefail

module load nvidia/cuda/13.0.0

JULIA="$HOME/.juliaup/bin/julia"
PROJECT_DIR="$HOME/repo759/PROJECT"
SCRIPT_REL="experiments/cuda_test.jl"

cd "$PROJECT_DIR"

echo "Running on: $(hostname)"
"$JULIA" --version
nvidia-smi

# Tell CUDA.jl to use the system CUDA toolkit (not the JLL precompiled on login node)
"$JULIA" --project=. -e 'using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'

"$JULIA" --project=. "$SCRIPT_REL"