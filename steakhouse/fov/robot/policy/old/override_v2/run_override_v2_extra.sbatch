#!/bin/bash
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --job-name=ovr_v2
#SBATCH --output=/scratch1/mishafu/steakhouse/carc_logs/ovr_v2_%A_%a.out

# Extra valid layouts to round out coverage (all construct cleanly).
LAYOUTS=(steak_gc02 steak_gc03 steak_gc04 steak_mid_1 steak_api)
LAYOUT=${LAYOUTS[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch1/mishafu/steakhouse
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=/home1/mishafu/.conda/envs/steakhouse-ai/bin/python

echo "=== task $SLURM_ARRAY_TASK_ID layout=$LAYOUT host=$(hostname) start=$(date) ==="
$PY -m fov.robot.policy.override_v2.carc_run "$LAYOUT" 120 10 "$ROOT/carc_results"
echo "=== task $SLURM_ARRAY_TASK_ID layout=$LAYOUT done=$(date) ==="
