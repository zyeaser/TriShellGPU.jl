#!/usr/bin/env bash
# =============================================================================
# submit_fem_all_gpus.sh
#
# Submits run_fem_solver.sh to every GPU type available on Euler.
# Each job runs independently; results are appended to the same CSV file.
#
# Usage (from PROJECT/ directory):
#   bash experiments/submit_fem_all_gpus.sh
#
# Monitor jobs:
#   squeue -u $USER
#
# Results land in:
#   experiments/benchmark_results_fem.csv
# =============================================================================

set -euo pipefail

SCRIPT="$HOME/repo759/PROJECT/experiments/run_fem_solver.sh"

echo "Submitting FEM solver benchmark to all GPU types on Euler..."
echo ""

# ── instruction partition ─────────────────────────────────────────────────────
echo -n "  rtx4000ada  (instruction) ... "
sbatch -p instruction \
       --gres=gpu:rtx4000ada:1 \
       -J fem_rtx4000ada \
       --time=00:30:00 \
       "$SCRIPT" rtx4000ada
# sbatch prints "Submitted batch job <id>" automatically

# ── research partition ────────────────────────────────────────────────────────
echo -n "  rtxa4500    (research)    ... "
sbatch -p research \
       --gres=gpu:rtxa4500:1 \
       -J fem_rtxa4500 \
       --time=00:30:00 \
       "$SCRIPT" rtxa4500

echo -n "  rtx2080ti   (research)    ... "
sbatch -p research \
       --gres=gpu:rtx2080ti:1 \
       -J fem_rtx2080ti \
       --time=00:30:00 \
       "$SCRIPT" rtx2080ti

echo -n "  a100        (research)    ... "
sbatch -p research \
       --gres=gpu:a100:1 \
       -J fem_a100 \
       --time=00:30:00 \
       "$SCRIPT" a100

echo -n "  h100        (research)    ... "
sbatch -p research \
       --gres=gpu:h100:1 \
       -J fem_h100 \
       --time=00:30:00 \
       "$SCRIPT" h100

# =============================================================================
echo ""
echo "All jobs submitted.  Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "Output logs:  fem_<gpu>_<jobid>.out / .err"
echo "Results CSV:  ~/repo759/PROJECT/experiments/benchmark_results_fem.csv"
