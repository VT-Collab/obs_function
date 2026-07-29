#!/bin/bash
#SBATCH --job-name=fovsearch
#SBATCH --array=0-30
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch1/mishafu/steakhouse/carc_logs/%A_%a.out

set -e
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate steakhouse-ai
export STEAK_ROOT=/scratch1/mishafu/steakhouse
export PYTHONPATH="$STEAK_ROOT:$PYTHONPATH"   # so `python script.py` finds the repo
cd "$STEAK_ROOT"
mkdir -p carc_results carc_logs fov/robot/policy/models

LAYOUTS=(steak_api steak_cram steak_cram2 steak_gc00 steak_gc01 steak_gc02 steak_gc03 \
steak_gc04 steak_gc05 steak_gc06 steak_gc07 steak_gs00 steak_gs02 steak_gs03 steak_gs04 \
steak_gs05 steak_gs06 steak_gs07 steak_gs08 steak_gs09 steak_island steak_island2 \
steak_mid_1 steak_mid_2 steak_none_3 steak_parrallel steak_side_2 steak_side_3 \
steak_side_4 steak_test steak_tshape)

L=${LAYOUTS[$SLURM_ARRAY_TASK_ID]}
W=fov/robot/policy/models/base_strong_${L}.pt   # 150-iter WELL-TRAINED baseline (fair)

echo "[$(date)] task $SLURM_ARRAY_TASK_ID layout=$L"
echo "training WELL-TRAINED baseline $L (150 iters) ..."
python -m fov.robot.policy.baseline.train 150 "$L" "$W" > carc_logs/train_${L}.log 2>&1 || echo "TRAIN FAIL $L"
echo "searching $L ..."
python fov/eval/search_layout.py "$L" "$W" 2 60 > carc_results/${L}.json 2> carc_logs/search_${L}.log || echo "SEARCH FAIL $L"
echo "[$(date)] DONE $L"
