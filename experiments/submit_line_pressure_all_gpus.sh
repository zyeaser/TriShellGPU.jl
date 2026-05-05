#!/usr/bin/env bash
# =============================================================================
# Submit the line-pressure GPU example to every GPU type on Euler.
#
# Usage from the project root:
#   bash experiments/submit_line_pressure_all_gpus.sh
#
# Monitor:
#   squeue -u "$USER"
# =============================================================================

set -euo pipefail

SCRIPT="$HOME/repo759/PROJECT/experiments/run_line_pressure_gpu.sh"

echo "Submitting line-pressure GPU example to all GPU types on Euler..."
echo ""

echo -n "  rtx4000ada  (instruction) ... "
sbatch -p instruction \
       --gres=gpu:rtx4000ada:1 \
       -J line_rtx4000ada \
       --time=00:15:00 \
       "$SCRIPT"

echo -n "  rtxa4500    (research)    ... "
sbatch -p research \
       --gres=gpu:rtxa4500:1 \
       -J line_rtxa4500 \
       --time=00:15:00 \
       "$SCRIPT"

echo -n "  rtx2080ti   (research)    ... "
sbatch -p research \
       --gres=gpu:rtx2080ti:1 \
       -J line_rtx2080ti \
       --time=00:15:00 \
       "$SCRIPT"

echo -n "  a100        (research)    ... "
sbatch -p research \
       --gres=gpu:a100:1 \
       -J line_a100 \
       --time=00:15:00 \
       "$SCRIPT"

echo -n "  h100        (research)    ... "
sbatch -p research \
       --gres=gpu:h100:1 \
       -J line_h100 \
       --time=00:15:00 \
       "$SCRIPT"

echo ""
echo "All line-pressure jobs submitted. Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "Output logs:"
echo "  line_pressure_<jobid>.out / line_pressure_<jobid>.err"
