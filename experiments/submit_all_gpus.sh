#!/usr/bin/env bash
# Submit benchmark jobs for all GPU types on Euler
# Run this from: ~/repo759/PROJECT/experiments/

SCRIPT="$HOME/repo759/PROJECT/experiments/run_gpu_benchmark.sh"

# RTX 4000 Ada — available on instruction partition
sbatch -p instruction --gres=gpu:rtx4000ada:1 -J rtx4000ada "$SCRIPT" rtx4000ada

# RTX A4500 — research partition only
sbatch -p research    --gres=gpu:rtxa4500:1   -J rtxa4500   "$SCRIPT" rtxa4500

# RTX 2080 Ti — research partition only
sbatch -p research    --gres=gpu:rtx2080ti:1  -J rtx2080ti  "$SCRIPT" rtx2080ti

# A100 — research partition only
sbatch -p research    --gres=gpu:a100:1       -J a100       "$SCRIPT" a100

# H100 — research partition only
sbatch -p research    --gres=gpu:h100:1       -J h100       "$SCRIPT" h100

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "Results will be written to:"
echo "  ~/repo759/PROJECT/experiments/benchmark_results.csv"
