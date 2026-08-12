#!/bin/bash -l
#SBATCH --job-name=qmdp_grid
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-119
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# The grid: 6 layouts x 5 cones x 4 PAIRINGS x 10 seeds = 2400 episodes.
#
# ONE TASK PER (layout, cone, PAIRING) -- 120 tasks, and that is a cost
# decision rather than a style one. The filter searches a branching frontier to
# depth 30 and computes a sight/pickup forecast per candidate, so it runs at
# 1.7-3.2 s/tick where the old enumerating filter ran at 0.08. Ten methods in
# one task, as this used to be, is ~30 h and does not fit any sane walltime;
# one pairing x ten seeds is 2-4 h, and a single failed cell re-runs alone.
#
# The two CONTROLS are deliberately NOT in this grid -- run them as their own
# job. qmdp-base must tie its baseline in every cell and qmdp-fixed is only
# interesting against qmdp, so neither belongs in a pairing column.
# (superseded comment follows)
# per (layout, cone) cell so all 30 cells run at once. Each task writes its own
# JSONL and report.py reads the whole directory at the end -- no shared file, so
# no interleaved-write corruption and a failed task can be re-run alone.
#
#   sbatch run_carc.sbatch                       the whole grid
#   sbatch --array=0-5 run_carc.sbatch           just the fov-30 column
#
# "-l" on the shebang is load-bearing: SLURM runs batch scripts as a NON-login
# shell, `module` is defined in the login profile, and without it `module load`
# fails silently and everything below runs against the system python.

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

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

# Keep results OFF home1 -- the quota there is tight and this writes a row per
# episode for 30 cells. QMDP_OUT names the grid, so two grids can be compared
# rather than one silently overwriting the other:
#
#   QMDP_OUT=/scratch1/$USER/qmdp_v2 sbatch run_carc.sbatch
#   python -m robot.filter.analysis.compare_grids OLD_DIR NEW_DIR
#
# That comparison is worth running on every change. Most edits in this package's
# history moved nothing at all, and the ones that did move something were not
# always the ones predicted to.
OUT=${QMDP_OUT:-/scratch1/$USER/qmdp}
mkdir -p "$OUT" logs

echo "=== task $SLURM_ARRAY_TASK_ID : $L fov$F methods=$M -> $OUT ==="
which python && python --version

# EVERY BASELINE HERE DRAWS ITS SUB-TASK, and that is the point of the method
# list rather than an accident of it. The filter consumes a DISTRIBUTION over
# sub-tasks; a policy that never draws has none, so subtask_dist.true_pi could
# only hand it a modelled Boltzmann. Scoring a filter fed a fiction of its
# baseline against a baseline that never held those preferences is not a paired
# comparison. Each qmdp-* row wraps the row it is scored against:
#
#     greedy  vs  qmdp-greedy        handoff  vs  qmdp
#     solo    vs  qmdp-solo          bayes    vs  qmdp-bayes
#
# report.PAIRS encodes exactly these.
python -m robot.filter.harness.evaluate \
    --layouts "$L" --fovs "$F" --seeds 0-9 \
    --methods "$M" \
    --horizon 400 \
    --out "$OUT/${TAG}_${L}_fov${F}.jsonl"

echo "=== done $L fov$F $TAG ==="
