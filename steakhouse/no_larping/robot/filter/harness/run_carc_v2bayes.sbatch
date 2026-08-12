#!/bin/bash -l
#SBATCH --job-name=qmdp_v2bayes
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-29
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# The v2-HUMAN arm for the BAYES pairing: 6 layouts x 5 cones x 10 seeds, bayes and
# qmdp-bayes, under common/tasks_containment.py.
#
# WHY THIS EXISTS SEPARATELY. The first v2 job ran handoff/qmdp only, and the
# pairing turns out to matter more than the layout: on chefs_table, qmdp beats
# handoff at all five cones, while qmdp-bayes LOSES to bayes at fov30 and fov360 and
# wins the three in the middle. bayes is the strong baseline -- at a wide cone it has
# the information already and does not need help -- so any claim about the human
# patch that rests on the handoff pairing alone says nothing about this one.
#
# ONLY THE V2 ARM IS RUN HERE. The v1 arm already exists -- job 10987487 in
# /scratch1/$USER/qmdp_v3 holds handoff and qmdp for all 30 cells at the same 10
# seeds under the same robot code -- so re-running it would burn an hour to
# reproduce numbers we have. Compare arm to arm with the same seeds, per cell.
#
#     v1 (current human)   /scratch1/$USER/qmdp_v3/{qmdp,}_<layout>_fov<F>.jsonl
#     v2 (this job)        /scratch1/$USER/qmdp_v2grid/<layout>_fov<F>.jsonl
#
# The robot code is identical in both arms and the patch routes on BeliefView, so
# every difference is the partner's doing. `handoff` is in both arms as the control:
# if IT moves, the patch changed the environment and the qmdp delta cannot be read
# on its own.

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

# ABSOLUTE INTERPRETER, NOT `conda activate`. Job 10998270 ran the BASE env despite
# an identical conda block, so gym was missing and all 16 tasks exited 0 in three
# seconds having written nothing -- a job that fails by SUCCEEDING, which neither
# squeue nor sacct flags. The guard at the end turns that failure mode into a
# non-zero exit.
PY=$HOME/.conda/envs/steakhouse-ai/bin/python
[ -x "$PY" ] || { echo "FATAL: no interpreter at $PY" >&2; exit 1; }

export STEAK_ROOT=$HOME/steakhouse
cd ~/steakhouse/no_larping

LAYOUTS=(back_bar banquet_pass butchery chefs_table divide pantry)
FOVS=(30 60 90 180 360)
L=${LAYOUTS[$(( SLURM_ARRAY_TASK_ID % 6 ))]}
F=${FOVS[$(( SLURM_ARRAY_TASK_ID / 6 ))]}

OUT=${QMDP_OUT:-/scratch1/$USER/qmdp_v2bayes}
mkdir -p "$OUT" logs

echo "=== task $SLURM_ARRAY_TASK_ID : $L fov$F (human v2) -> $OUT ==="
echo "interpreter: $PY"; "$PY" --version

"$PY" -m robot.filter.dev.eval_human_v2 \
    --layouts "$L" --fovs "$F" --seeds 0-9 \
    --methods bayes,qmdp-bayes \
    --horizon 400 \
    --out "$OUT/${L}_fov${F}.jsonl"

OUTFILE="$OUT/${L}_fov${F}.jsonl"
[ -s "$OUTFILE" ] || { echo "FATAL: $OUTFILE is empty" >&2; exit 1; }
echo "=== done $L fov$F -- $(wc -l < "$OUTFILE") rows ==="
