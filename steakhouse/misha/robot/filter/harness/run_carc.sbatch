#!/bin/bash -l
#SBATCH --job-name=fov_c8_grid
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-119
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# The grid: 6 layouts x 5 cones x 4 PAIRINGS x 20 seeds = 4800 episodes.
# Mirrors no_larping/robot/filter/harness/run_carc.sbatch's structure, retargeted
# at this package's own FoV-only cap/decay filter (robot/filter/core/fov_filter.py)
# at cap=8, self-contained under ~/steakhouse/misha -- no dependency on
# no_larping at runtime.
#
# ONE TASK PER (layout, cone, PAIRING) -- 120 tasks, same cost reasoning as the
# qmdp grid this is modelled on: a single failed cell re-runs alone rather than
# losing a whole multi-method task.
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
cd ~/steakhouse/misha

LAYOUTS=(back_bar banquet_pass butchery chefs_table divide pantry)
FOVS=(30 60 90 180 360)

# Each fov-*-c8 row wraps the baseline immediately to its left -- see
# robot/methods.py's fov-greedy-c8/fov-solo-c8/fov-c8/fov-bayes-c8 rows, cap=8.0
# fixed, fov_decay left at its default (0.5) on every row.
METHODS=("greedy,fov-greedy-c8" "solo,fov-solo-c8" "handoff,fov-c8" "bayes,fov-bayes-c8")

CELL=$(( SLURM_ARRAY_TASK_ID % 30 ))
MI=$(( SLURM_ARRAY_TASK_ID / 30 ))
L=${LAYOUTS[$(( CELL % 6 ))]}
F=${FOVS[$(( CELL / 6 ))]}
M=${METHODS[$MI]}
TAG=$(echo "$M" | cut -d, -f2)

# Keep results OFF home1 -- the quota there is tight and this writes a row per
# episode for 30 cells. FOV_OUT names the grid, so two grids can be compared
# rather than one silently overwriting the other:
#
#   FOV_OUT=/scratch1/$USER/fov_c8_v2 sbatch run_carc.sbatch
OUT=${FOV_OUT:-/scratch1/$USER/fov_c8}
mkdir -p "$OUT" logs

echo "=== task $SLURM_ARRAY_TASK_ID : $L fov$F methods=$M -> $OUT ==="
which python && python --version

# EVERY BASELINE HERE DRAWS ITS SUB-TASK -- see robot/methods.py's module
# docstring for why that is a correctness requirement rather than a style
# choice. Each fov-*-c8 row wraps the row it is scored against:
#
#     greedy  vs  fov-greedy-c8        handoff  vs  fov-c8
#     solo    vs  fov-solo-c8          bayes    vs  fov-bayes-c8
#
# analysis/report.PAIRS and analysis/results_tables.COLS encode exactly these.
#---- CHANGED: capture and act on the exit code explicitly. This script has
#     no `set -e` (module/conda init scripts can return nonzero internally
#     for benign reasons, and a blanket errexit would make THOSE kill the
#     job) -- but that means a real failure here used to fall straight
#     through to "echo done" and exit 0, so SLURM/sacct reported COMPLETED
#     for a task that never wrote a single row. Bit us once already: every
#     task in a prior run "completed" while crashing on the first line on
#     an AssertionError. Never again silently -- a nonzero exit here now
#     propagates as the job step's own exit code.
#----------------------------------------------------------------------
python -m robot.filter.harness.evaluate \
    --layouts "$L" --fovs "$F" --seeds 0-19 \
    --methods "$M" \
    --horizon 400 \
    --out "$OUT/${TAG}_${L}_fov${F}.jsonl"
RC=$?

if [ $RC -ne 0 ]; then
    echo "=== FAILED (exit $RC) $L fov$F $TAG ===" >&2
    exit $RC
fi

echo "=== done $L fov$F $TAG ==="
