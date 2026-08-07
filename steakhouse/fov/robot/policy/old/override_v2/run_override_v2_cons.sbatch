#!/bin/bash
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --array=0-8
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=ovr_c
#SBATCH --output=/scratch1/mishafu/steakhouse/carc_logs/ovr_c_%A_%a.out

# CONSERVATIVE override (baseline owns the task; module only reshapes the path
# through the cone when sighted+confident) + longer horizon so all 4 orders
# actually complete and TIME-to-finish becomes a discriminating metric.
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
# args: layout iters n_eps outdir mode horizon
$PY -m fov.robot.policy.override_v2.carc_run "$LAYOUT" 120 10 \
    "$ROOT/carc_results_cons" conservative 450
echo "=== task $SLURM_ARRAY_TASK_ID layout=$LAYOUT done=$(date) ==="
