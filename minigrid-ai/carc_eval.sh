#!/bin/bash -l
# Evaluate every trained baseline checkpoint, one array task per checkpoint.
# Each task appends a tab-separated line to eval_results/ for collation.
#
#   sbatch carc_eval.sh
#
#SBATCH --job-name=nofov_eval
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-24
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate minigrid

cd ~/minigrid-ai
export PYTHONPATH=$PWD/Minigrid:$PWD
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p eval_results logs
CKPTS=($(ls robot/policy/neural/baseline/no_fov/checkpoints/sweep*/*.pt 2>/dev/null | sort))
N=${#CKPTS[@]}
IDX=$((SLURM_ARRAY_TASK_ID-1))

if [ "$IDX" -ge "$N" ]; then
  echo "no checkpoint at index $IDX (only $N present) - nothing to do"
  exit 0
fi

CK=${CKPTS[$IDX]}
NAME=$(basename "$CK" .pt)
echo "evaluating $CK"

# task 1 also emits the no-assist reference row
EXTRA=""
[ "$SLURM_ARRAY_TASK_ID" = "1" ] && EXTRA="--with-noassist"

python carc_eval.py "$CK" --seeds 25 $EXTRA > eval_results/$NAME.tsv
echo "wrote eval_results/$NAME.tsv"
cat eval_results/$NAME.tsv
