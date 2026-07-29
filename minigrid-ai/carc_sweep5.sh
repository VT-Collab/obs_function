#!/bin/bash -l
# Sweep 5: DENSE GEOMETRIC SHAPING re-enabled (user-approved).
# Evidence: the OLD 22-ch baseline was genuinely STATE-DEPENDENT (4 actions,
# dominant 33.7%, P(speak)spread 0.268) while every post-cleanup checkpoint is a
# CONSTANT (spread 0.0000). The one signal the old setup had and the new one lacks
# is dense per-step shaping (it ran silently at 1.0 via the falsy-default bug).
# Now implemented correctly: F = gamma*PHI(s') - PHI(s), PHI = -griddist to the next
# needed thing. Pure geometry - no knowledge base, no FOV.
# g04 deliberately replicates the ORIGINAL recipe (fpu=512, lr=1e-3, shaping=1.0).
#SBATCH --job-name=nofov_shape
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=1-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -e
# Call the env's interpreter by ABSOLUTE PATH. `conda activate` failed silently on
# the compute nodes when several array jobs activated at once (NFS contention): the
# job fell back to the module python and died with ModuleNotFoundError: torch.
PY_BIN=$HOME/.conda/envs/minigrid/bin/python
$PY_BIN -c "import torch, torch_ac, gymnasium" || { echo "FATAL: env broken"; exit 1; }
cd ~/minigrid-ai
export PYTHONPATH=$PWD/Minigrid:$PWD SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
FRAMES=400000
CK=robot/policy/neural/baseline/no_fov/checkpoints/sweep5
mkdir -p $CK logs
# name shaping method   fpu   lr   ent    comm
CONFIGS=(
  "g01 1.0 rec_ppo 4096 3e-4 0.0005 0.002"
  "g02 0.5 rec_ppo 4096 3e-4 0.0005 0.002"
  "g03 1.0 ppo     4096 3e-4 0.0005 0.002"
  "g04 1.0 rec_ppo  512 1e-3 0.0005 0.002"
  "g05 0.2 rec_ppo 4096 3e-4 0.0005 0.002"
  "g06 1.0 rec_ppo 4096 3e-4 0.0005 0.005"
)
read -r NAME SHAPE METHOD FPU LR ENT COMM <<< "${CONFIGS[$((SLURM_ARRAY_TASK_ID-1))]}"
echo "=== $NAME | shaping=$SHAPE method=$METHOD fpu=$FPU lr=$LR ent=$ENT comm=$COMM key=0.30 warmup=0.5 ==="
$PY_BIN -m robot.policy.neural.baseline.no_fov.train \
    --method "$METHOD" --seed 1 --frames "$FRAMES" \
    --shaping "$SHAPE" --frames-per-update "$FPU" --ppo-lr "$LR" \
    --entropy-start "$ENT" --entropy-end "$ENT" \
    --comm-cost "$COMM" --key-bonus 0.30 --comm-warmup-frac 0.5 \
    --no-wandb --save "$CK/$NAME.pt"
echo "=== $NAME done ==="
