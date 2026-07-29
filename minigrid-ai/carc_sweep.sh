#!/bin/bash -l
# Baseline hyperparameter sweep for the no_fov raw-state baseline.
#
# Goal: find a config where the LEARNED baseline is genuinely helpful (beats a
# silent robot) without any hard-coded assistance logic. Only hyperparameters
# and training schedules vary - the env, the reward shape and the unguarded
# action space are identical across every task.
#
#   sbatch carc_sweep.sh
#   squeue -u $USER
#
#SBATCH --job-name=nofov_sweep
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=1-12
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate minigrid

cd ~/minigrid-ai
export PYTHONPATH=$PWD/Minigrid:$PWD
export SDL_VIDEODRIVER=dummy          # pygame is imported but never rendered
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

FRAMES=1000000
CK=robot/policy/neural/baseline/no_fov/checkpoints/sweep
mkdir -p $CK logs

# name  warmup  ent_start  ent_end  key_bonus  lr      seed
CONFIGS=(
  "c01  0.5  0.0005  0.0005  0.15  1e-3  1"
  "c02  0.5  0.0005  0.0005  0.30  1e-3  1"
  "c03  0.3  0.0010  0.0002  0.15  1e-3  1"
  "c04  0.7  0.0010  0.0002  0.15  1e-3  1"
  "c05  0.5  0.0020  0.0005  0.15  1e-3  1"
  "c06  0.5  0.0005  0.0005  0.15  3e-4  1"
  "c07  0.3  0.0005  0.0005  0.15  1e-3  1"
  "c08  0.0  0.0005  0.0005  0.15  1e-3  1"   # no-warmup control
  "c09  0.5  0.0002  0.0002  0.15  1e-3  1"
  "c10  0.5  0.0010  0.0010  0.15  1e-3  1"
  "c11  0.5  0.0005  0.0005  0.15  1e-3  2"   # seed replicate of c01
  "c12  0.5  0.0005  0.0005  0.60  1e-3  1"
)

read -r NAME WARMUP ENT_S ENT_E KEY LR SEED <<< "${CONFIGS[$((SLURM_ARRAY_TASK_ID-1))]}"

echo "=== $NAME | warmup=$WARMUP ent=$ENT_S->$ENT_E key=$KEY lr=$LR seed=$SEED frames=$FRAMES ==="

python -m robot.policy.neural.baseline.no_fov.train \
    --method rec_ppo --seed "$SEED" --frames "$FRAMES" \
    --comm-warmup-frac "$WARMUP" \
    --entropy-start "$ENT_S" --entropy-end "$ENT_E" \
    --key-bonus "$KEY" --ppo-lr "$LR" \
    --no-wandb --save "$CK/$NAME.pt"

echo "=== $NAME done ==="
