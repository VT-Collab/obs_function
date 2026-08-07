#!/bin/bash -l
#SBATCH --job-name=fov_rl2
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
  echo "############ $L ############"
  # BASELINE: zero FOV code involved (verified by deleting inference/ + module/)
  python -m fov.robot.policy.baseline.train 120 $L base_$L.pt
  # MODULE: same weights, FOV logit bias only
  python -m fov.robot.policy.end_to_end.compare base_$L.pt $L 40
done
