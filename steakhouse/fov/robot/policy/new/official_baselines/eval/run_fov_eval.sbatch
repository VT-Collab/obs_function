#!/bin/bash -l
# ===========================================================================
# THE BASELINE TABLE.  Every trained policy x every FOV, played against the
# LimitedVisionSteakHuman.
# ===========================================================================
# One array task per LAYOUT. Each task sweeps all algos x all their seeds x all
# six FOVs x --episodes episodes, and appends JSONL rows to a per-layout file.
# aggregate.py turns the pile of rows into the table.
#
# Per task, with the default 20 episodes:
#     (11 sp + 5 e3t + 5 sp_eps + 2 floors) = 23 policies
#     23 x 6 fovs x 20 episodes = 2,760 episodes
# The cost is the HUMAN, not the robot -- LimitedVisionSteakHuman re-scans its
# cone and re-runs a BFS over seen floor every single tick. The 5x5 conv is
# microseconds. That is also why torch is pinned to 1 thread inside the script:
# extra threads only add contention to a python-bound loop.
#
# DO NOT RUN THIS UNTIL TRAINING HAS FINISHED. It reads final.pt, which is
# written once at the very end of a run. Check first:
#     ls /scratch1/$USER/steakhouse_zsc/sp/*/final.pt | wc -l    # want 77
#
# WHY EPISODES MATTER MORE HERE THAN IN A SELF-PLAY EVAL: a DNF is scored at
# horizon+100, so a single non-finish moves a mean by ~100 ticks. The first
# 3-episode probe on gc00 had exactly that problem. 20 per (policy, fov) with
# 11 seeds means the SP row is a mean over 220 episodes.

#SBATCH --job-name=fov_eval
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse

LAYOUTS=(steak_gc00 steak_gc03 steak_gc04 steak_gc06 steak_cram steak_cram2 steak_gs00)
LAYOUT="${LAYOUTS[$SLURM_ARRAY_TASK_ID]}"

EPISODES="${EPISODES:-20}"
MODE="${MODE:-sample}"
ALGOS="${ALGOS:-noop,random,sp,sp_eps,e3t}"
OUTDIR="${OUTDIR:-/scratch1/$USER/steakhouse_zsc/eval}"
mkdir -p "$OUTDIR"

# COLLISIONS=0 makes the agents pass through each other, isolating partner
# observability from physical obstruction. Read disable_collisions() in
# fov_human_eval.py before reporting anything from that arm. The two arms write
# to DIFFERENT filenames and every row carries a "collisions" field, so they
# cannot be pooled by accident.
COLLISIONS="${COLLISIONS:-1}"
if [ "$COLLISIONS" = "0" ]; then
  COLL_ARG="--no_collisions"; COLL_TAG="nocoll"
else
  COLL_ARG=""; COLL_TAG="coll"
fi

# append-only rows file per layout. deleting it is how you re-run cleanly --
# the harness appends so a partial run is never silently mixed with a full one
OUT="$OUTDIR/${LAYOUT}_${MODE}_${COLL_TAG}.jsonl"

echo "######## STAGING LAYOUTS ########"
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/ 2>/dev/null || true

echo "######## WHAT EXISTS ########"
for a in sp e3t sp_eps; do
  n=$(ls /scratch1/$USER/steakhouse_zsc/$a/${LAYOUT}_seed*/final.pt 2>/dev/null | wc -l)
  echo "  $a: $n finished runs for $LAYOUT"
done

echo "######## EVAL ########"
echo "layout=$LAYOUT episodes=$EPISODES mode=$MODE algos=$ALGOS"
echo "out=$OUT"

python fov/robot/policy/new/official_baselines/eval/fov_human_eval.py \
    --layout "$LAYOUT" \
    --algos "$ALGOS" \
    --all_fovs \
    --episodes "$EPISODES" \
    --mode "$MODE" \
    $COLL_ARG \
    --out "$OUT"

echo "######## DONE ########"
wc -l "$OUT"
