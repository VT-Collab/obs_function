#!/bin/bash
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --job-name=ovr_v2
#SBATCH --output=/scratch1/mishafu/steakhouse/carc_logs/ovr_v2_%A_%a.out

# One layout per array task. Contention layouts (gc00/cram/cram2) are the best
# testbed per the layout study; gc01 and plain steak included as contrast.
LAYOUTS=(steak_gc00 steak_cram steak_cram2 steak_gc01 steak)
LAYOUT=${LAYOUTS[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch1/mishafu/steakhouse
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
# keep BLAS/OpenMP from oversubscribing threads (the login-node failure mode)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=/home1/mishafu/.conda/envs/steakhouse-ai/bin/python

echo "=== task $SLURM_ARRAY_TASK_ID layout=$LAYOUT host=$(hostname) start=$(date) ==="
$PY -m fov.robot.policy.override_v2.carc_run "$LAYOUT" 120 10 "$ROOT/carc_results"
echo "=== task $SLURM_ARRAY_TASK_ID layout=$LAYOUT done=$(date) ==="
