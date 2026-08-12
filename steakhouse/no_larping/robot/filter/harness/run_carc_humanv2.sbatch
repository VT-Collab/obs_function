#!/bin/bash -l
#SBATCH --job-name=qmdp_humanv2
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-15
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# A/B on the EXPERIMENTAL human redundancy patch (common/tasks_containment.py).
#
# The question: how much of the qmdp gain in RESULTS.md section 2c is the filter
# exploiting a partner that over-produces? The patch makes the human's redundancy
# check count composites -- a garnish_dish is a garnish AND a plate -- so it stops
# fetching onions and plates it already has. If the gain survives, the patch is
# worth merging and measuring properly. If it vanishes, the gain was an artefact of
# a bug in the partner.
#
# EIGHT CELLS, chosen as the ones where section 2c showed the largest gains, because
# that is where an artefact would be hiding. Not a full grid -- this is a screen, and
# whichever way it comes out the answer is "now run the full thing".
#
#   cell x {v1 human, v2 human}  = 16 tasks, each handoff + qmdp over 5 seeds.
#
# Both arms run the SAME robot code; only the human's redundancy check differs. The
# baselines are in both arms on purpose: they are the control, and if `handoff` moves
# between arms then the patch changed the environment and the qmdp delta cannot be
# read on its own.

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

# CALL THE ENV'S INTERPRETER BY PATH, and do not trust `conda activate`. The first
# submission of this job silently ran the BASE env -- `which python` reported
# /apps/conda/miniforge3/.../bin/python, gym was missing, and all 16 tasks exited 0
# in three seconds having written nothing. The identical conda block in
# run_carc.sbatch activated correctly for job 10987487, so activation is flaky here
# rather than wrong -- and a job that fails by SUCCEEDING is the worst kind, because
# nothing in squeue or sacct says anything is wrong. An absolute path cannot
# half-work, and the guard at the end turns a silent no-op into a failed task.
PY=$HOME/.conda/envs/steakhouse-ai/bin/python
[ -x "$PY" ] || { echo "FATAL: no interpreter at $PY" >&2; exit 1; }

export STEAK_ROOT=$HOME/steakhouse
cd ~/steakhouse/no_larping

CELLS=("back_bar 180" "divide 180" "pantry 30" "chefs_table 90" \
       "butchery 360" "divide 90" "back_bar 60" "banquet_pass 30")

CI=$(( SLURM_ARRAY_TASK_ID % 8 ))
ARM=$(( SLURM_ARRAY_TASK_ID / 8 ))
set -- ${CELLS[$CI]}
L=$1
F=$2

OUT=${QMDP_OUT:-/scratch1/$USER/qmdp_humanv2}
mkdir -p "$OUT" logs

if [ "$ARM" -eq 0 ]; then
    MOD=robot.filter.harness.evaluate
    TAG=v1
else
    MOD=robot.filter.dev.eval_human_v2
    TAG=v2
fi

echo "=== task $SLURM_ARRAY_TASK_ID : $L fov$F arm=$TAG ($MOD) -> $OUT ==="
echo "interpreter: $PY"; "$PY" --version

"$PY" -m "$MOD" \
    --layouts "$L" --fovs "$F" --seeds 0-4 \
    --methods handoff,qmdp \
    --horizon 400 \
    --out "$OUT/${TAG}_${L}_fov${F}.jsonl"

OUTFILE="$OUT/${TAG}_${L}_fov${F}.jsonl"
[ -s "$OUTFILE" ] || { echo "FATAL: $OUTFILE is empty" >&2; exit 1; }
echo "=== done $TAG $L fov$F -- $(wc -l < "$OUTFILE") rows ==="
