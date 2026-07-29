#!/bin/bash -l
# Sweep 3: attack the CONSTANT-POLICY COLLAPSE.
# Every config in sweeps 1-2 converged to a state-INDEPENDENT policy (d12/d04 emit
# 'key' 100% of steps at prob 1.000). Hypothesis: frames_per_update=512 with one env
# is ~2.7 episodes/update, so advantages are noise and PPO ratchets into a constant.
# Fix = much bigger rollouts + lower lr. Also tries feed-forward (no GRU).
#SBATCH --job-name=nofov_fix
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=1-8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate minigrid
cd ~/minigrid-ai
export PYTHONPATH=$PWD/Minigrid:$PWD SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
FRAMES=400000
CK=robot/policy/neural/baseline/no_fov/checkpoints/sweep3
mkdir -p $CK logs
# name method   fpu   lr    ent_s  ent_e  comm   key  warmup
CONFIGS=(
  "e01 rec_ppo 4096 3e-4 0.003 0.003 0.002 0.30 0.5"
  "e02 rec_ppo 4096 1e-4 0.003 0.003 0.002 0.30 0.5"
  "e03 rec_ppo 2048 3e-4 0.003 0.003 0.002 0.30 0.5"
  "e04 ppo     4096 3e-4 0.003 0.003 0.002 0.30 0.5"
  "e05 ppo     4096 1e-4 0.003 0.003 0.002 0.30 0.5"
  "e06 rec_ppo 4096 3e-4 0.010 0.001 0.002 0.30 0.5"
  "e07 rec_ppo 8192 3e-4 0.003 0.003 0.002 0.30 0.5"
  "e08 ppo     4096 3e-4 0.003 0.003 0.005 0.15 0.3"
)
read -r NAME METHOD FPU LR ENT_S ENT_E COMM KEY WARMUP <<< "${CONFIGS[$((SLURM_ARRAY_TASK_ID-1))]}"
echo "=== $NAME | method=$METHOD fpu=$FPU lr=$LR ent=$ENT_S->$ENT_E comm=$COMM key=$KEY warmup=$WARMUP ==="
python -m robot.policy.neural.baseline.no_fov.train \
    --method "$METHOD" --seed 1 --frames "$FRAMES" \
    --frames-per-update "$FPU" --ppo-lr "$LR" \
    --entropy-start "$ENT_S" --entropy-end "$ENT_E" \
    --comm-cost "$COMM" --key-bonus "$KEY" --comm-warmup-frac "$WARMUP" \
    --no-wandb --save "$CK/$NAME.pt"
echo "=== $NAME done ==="
