#!/bin/bash -l
# ===========================================================================
# QMDP FOV module -- one slurm array task per config in a named grid.
#
#   GRID=beta    sbatch --array=0-3    run_qmdp.sbatch
#   GRID=weights sbatch --array=0-23%12 run_qmdp.sbatch
#   GRID=ablate  sbatch --array=0-76%16 run_qmdp.sbatch
#   GRID=final   sbatch --array=0-10%11 run_qmdp.sbatch
#
# The exact array size for a grid:
#   python fov/robot/policy/old/module/QMDP/configs.py <grid> --n
#
# Everything is CPU-bound python (the mdp, the human's BFS, the shadow cones);
# torch does microseconds of conv on a 5x5..15x10 grid. 2 cpus is plenty and
# more concurrent tasks beats more cores per task -- same conclusion
# official_baselines/SP/CARC_RUNS.md reached for training.
# ===========================================================================
#SBATCH --job-name=qmdp
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=qmdp_%x_%A_%a.out
#SBATCH --error=qmdp_%x_%A_%a.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse

QMDP=fov/robot/policy/old/module/QMDP
GRID="${GRID:-final}"
OUTDIR="${OUTDIR:-/scratch1/$USER/steakhouse_qmdp/$GRID}"
mkdir -p "$OUTDIR"

# from_layout_name() only reads overcooked_ai_py/data/layouts. cp -n so
# concurrent array tasks cannot clobber each other. (_paths.stage_layouts()
# does the same thing from python; this is belt and braces.)
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/ 2>/dev/null || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export STEAK_SP_CKPT_ROOT="/scratch1/$USER/steakhouse_sp/specialist"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
OUT="$OUTDIR/task_${TASK}.jsonl"
rm -f "$OUT"

echo "######## $(date) grid=$GRID task=$TASK out=$OUT ########"
python -c "import torch,numpy,sys;print('python',sys.version.split()[0],'torch',torch.__version__,'numpy',numpy.__version__)"

ARGS=$(cd $QMDP && python configs.py "$GRID" --args "$TASK" --out "$OUT")
echo "ARGS: $ARGS"

cd $QMDP
python evaluate.py $ARGS

echo "######## DONE $(date) ########"
wc -l "$OUT"
