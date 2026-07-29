#!/bin/bash -l
# MISHA NEW CHANGE - train the FOV-blind baseline robot with PPO, then evaluate
# the FOV module (map_fov + entropy logit bias) on the SAME weights.
# One robot + one limited-vision human, shared team reward, human FOV resampled
# every episode so the baseline must learn the best strategy under uncertainty.
#SBATCH --job-name=fov_rl
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse
python -c "import torch; print('torch', torch.__version__)"
for L in steak_side_2 steak_island steak_api; do
  echo "############ LAYOUT $L ############"
  python -m fov.robot.policy.end_to_end.train_and_compare 120 $L
done
