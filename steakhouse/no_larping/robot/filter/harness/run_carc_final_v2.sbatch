#!/bin/bash -l
#SBATCH --job-name=qmdp_final_v2
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-119
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# FINAL SWEEP, v2 HUMAN: 6 layouts x 5 cones x 4 pairings x 10 seeds = 2400 episodes.
#
#   v1  the shipped human model            robot.filter.harness.evaluate
#   v2  composites count toward saturation robot.filter.dev.eval_human_v2
#
# The two arms differ ONLY in the human's redundancy check. Same robot code, same
# layouts, same seeds, so the pair of tables is a controlled comparison and the
# baselines in each are the control on everything else.
#
# WHY THIS SUPERSEDES EVERY EARLIER GRID. Five of the six .layout files were edited
# mid-experiment, and CARC was still holding the pre-edit copies -- the grid in
# RESULTS.md section 2c recorded fingerprints 46fec8e416 / 57da296901 / 47631e1490 /
# c9d93b6640 / faeee8fb24 for banquet_pass, butchery, chefs_table, divide and pantry,
# and the current files hash to 252a7bcc0d / a86a1d541b / 0aec924651 / 57e9e2bdf5 /
# 707da824f6. Those numbers describe kitchens that no longer exist. layout/ is now
# synced and verified; every row here carries its fingerprint, so check them before
# quoting anything next to an older grid.
#
# ONE TASK PER (layout, cone, pairing). The filter runs at 0.9-3.2 s/tick, so 20
# episodes is 1-2 h; putting all four pairings in one task would not fit a walltime.

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

# ABSOLUTE INTERPRETER. Job 10998270 silently ran the BASE env despite an identical
# conda block -- gym missing, all 16 tasks exited 0 in three seconds having written
# nothing. A job that fails by SUCCEEDING is invisible in squeue and sacct, so the
# path is explicit and the guard at the end makes an empty output a failure.
PY=$HOME/.conda/envs/steakhouse-ai/bin/python
[ -x "$PY" ] || { echo "FATAL: no interpreter at $PY" >&2; exit 1; }

export STEAK_ROOT=$HOME/steakhouse
cd ~/steakhouse/no_larping

LAYOUTS=(back_bar banquet_pass butchery chefs_table divide pantry)
FOVS=(30 60 90 180 360)
METHODS=("handoff,qmdp" "greedy,qmdp-greedy" "solo,qmdp-solo" "bayes,qmdp-bayes")

CELL=$(( SLURM_ARRAY_TASK_ID % 30 ))
MI=$(( SLURM_ARRAY_TASK_ID / 30 ))
L=${LAYOUTS[$(( CELL % 6 ))]}
F=${FOVS[$(( CELL / 6 ))]}
M=${METHODS[$MI]}
TAG=$(echo "$M" | cut -d, -f2)

OUT=${QMDP_OUT:-/scratch1/$USER/qmdp_final_v2}
mkdir -p "$OUT" logs

echo "=== task $SLURM_ARRAY_TASK_ID : $L fov$F methods=$M arm=v2 -> $OUT ==="
echo "interpreter: $PY"; "$PY" --version

"$PY" -m robot.filter.dev.eval_human_v2 \
    --layouts "$L" --fovs "$F" --seeds 0-9 \
    --methods "$M" \
    --horizon 400 \
    --out "$OUT/${TAG}_${L}_fov${F}.jsonl"

OUTFILE="$OUT/${TAG}_${L}_fov${F}.jsonl"
[ -s "$OUTFILE" ] || { echo "FATAL: $OUTFILE is empty" >&2; exit 1; }
echo "=== done v2 $L fov$F $TAG -- $(wc -l < "$OUTFILE") rows ==="
