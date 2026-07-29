#!/bin/bash -l
# Sweep 4: the combination sweep 3 missed - BIG ROLLOUTS + LOW ENTROPY.
# sweeps 1-2: fpu=512 (noisy advantages) + ent=0.0005 -> collapsed to a constant.
# sweep 3:    fpu=4096 (clean advantages) + ent=0.003 -> entropy bonus ~1.02/episode,
#             comparable to the whole +1 prize, so the policy never commits.
# Here: fpu=4096 AND ent~0.0005 (bonus ~0.17 of the prize). Hyperparameters only.
#SBATCH --job-name=nofov_fix2
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=1-6
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
CK=robot/policy/neural/baseline/no_fov/checkpoints/sweep4
mkdir -p $CK logs
# name method   fpu   lr    ent
CONFIGS=(
  "f01 rec_ppo 4096 3e-4 0.0005"
  "f02 rec_ppo 4096 1e-4 0.0005"
  "f03 ppo     4096 3e-4 0.0005"
  "f04 ppo     4096 1e-4 0.0005"
  "f05 rec_ppo 4096 3e-4 0.0010"
  "f06 ppo     4096 3e-4 0.0010"
)
read -r NAME METHOD FPU LR ENT <<< "${CONFIGS[$((SLURM_ARRAY_TASK_ID-1))]}"
echo "=== $NAME | method=$METHOD fpu=$FPU lr=$LR ent=$ENT comm=0.002 key=0.30 warmup=0.5 ==="
python -m robot.policy.neural.baseline.no_fov.train \
    --method "$METHOD" --seed 1 --frames "$FRAMES" \
    --frames-per-update "$FPU" --ppo-lr "$LR" \
    --entropy-start "$ENT" --entropy-end "$ENT" \
    --comm-cost 0.002 --key-bonus 0.30 --comm-warmup-frac 0.5 \
    --no-wandb --save "$CK/$NAME.pt"
echo "=== $NAME done ==="
