#!/bin/bash -l
# Sweep 2: REWARD MAGNITUDES, not schedules.
#
# Sweep 1 varied warmup/entropy/lr and everything hovered around the mute score
# (~+0.10). The gap a perfect assistant can earn is only ~0.22 at comm=0.005,
# against per-episode noise of sigma~0.47 - too thin to learn from. So widen it:
#   - cheaper words   -> exploring speech costs almost nothing, so the policy can
#                        find out that key/door/goal help and dead_room sabotages
#   - louder key bonus-> pays the robot's main causal lever (steering them to the
#                        key that actually opens the goal room)
#
# Still no cheating: these are reward NUMBERS and training schedules. No FOV, no
# knowledge-base reads, no action masking, no hard-coded assistance logic.
# d12 is the control - words completely free. If that one still will not speak,
# the problem was never the price.
#
#   sbatch carc_sweep2.sh
#
#SBATCH --job-name=nofov_rew
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
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

FRAMES=1000000
CK=robot/policy/neural/baseline/no_fov/checkpoints/sweep2
mkdir -p $CK logs

# name  comm    key   warmup  ent_s   ent_e   lr    seed
CONFIGS=(
  "d01  0.001  0.30  0.5  0.0005  0.0005  1e-3  1"
  "d02  0.001  0.50  0.5  0.0005  0.0005  1e-3  1"
  "d03  0.002  0.30  0.5  0.0005  0.0005  1e-3  1"
  "d04  0.002  0.50  0.5  0.0005  0.0005  1e-3  1"
  "d05  0.001  0.30  0.3  0.0005  0.0005  1e-3  1"
  "d06  0.001  0.50  0.7  0.0005  0.0005  1e-3  1"
  "d07  0.0005 0.50  0.5  0.0005  0.0005  1e-3  1"
  "d08  0.001  1.00  0.5  0.0005  0.0005  1e-3  1"
  "d09  0.001  0.30  0.5  0.0002  0.0002  1e-3  1"
  "d10  0.001  0.50  0.5  0.0002  0.0002  3e-4  1"
  "d11  0.001  0.50  0.5  0.0005  0.0005  1e-3  2"
  "d12  0.000  0.50  0.0  0.0005  0.0005  1e-3  1"   # free-speech control
)

read -r NAME COMM KEY WARMUP ENT_S ENT_E LR SEED <<< "${CONFIGS[$((SLURM_ARRAY_TASK_ID-1))]}"

echo "=== $NAME | comm=$COMM key=$KEY warmup=$WARMUP ent=$ENT_S->$ENT_E lr=$LR seed=$SEED frames=$FRAMES ==="

python -m robot.policy.neural.baseline.no_fov.train \
    --method rec_ppo --seed "$SEED" --frames "$FRAMES" \
    --comm-cost "$COMM" --key-bonus "$KEY" \
    --comm-warmup-frac "$WARMUP" \
    --entropy-start "$ENT_S" --entropy-end "$ENT_E" \
    --ppo-lr "$LR" \
    --no-wandb --save "$CK/$NAME.pt"

echo "=== $NAME done ==="
