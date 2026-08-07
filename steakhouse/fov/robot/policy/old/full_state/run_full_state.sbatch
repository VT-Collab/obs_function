#!/bin/bash
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --array=0-8
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=02:30:00
#SBATCH --job-name=fs_cmp
#SBATCH --output=/scratch1/mishafu/steakhouse/carc_logs/fs_%A_%a.out

# Observation comparison: old-17dim vs full-state flat vs full-state grid (CNN),
# all FOV-blind, one layout per task.
LAYOUTS=(steak_gc00 steak_cram steak_cram2 steak_gc01 steak_gc02 steak_gc03 steak_gc04 steak_mid_1 steak_api)
LAYOUT=${LAYOUTS[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch1/mishafu/steakhouse
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=/home1/mishafu/.conda/envs/steakhouse-ai/bin/python

echo "=== task $SLURM_ARRAY_TASK_ID layout=$LAYOUT host=$(hostname) start=$(date) ==="
# args: layout iters n_eps outdir horizon
$PY -m fov.robot.policy.full_state.carc_run "$LAYOUT" 120 10 "$ROOT/carc_results_fs" 450
echo "=== task $SLURM_ARRAY_TASK_ID layout=$LAYOUT done=$(date) ==="
