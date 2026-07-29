#!/bin/bash -l
# MISHA NEW CHANGE - validate DESIGNED layouts (geometry solved, not searched).
# Random search: 0/681 passed, mean real divergence 0.1, because the
# discriminating region is 2-9 cells of ~90. Designed placement hit both gates on
# the first layout built: all FOVs DONE in 129 steps with 12 subtasks, full-vision
# control 0% stalls, and 2/3 FOV pairs showing real late-half divergence.
#SBATCH --job-name=fov_designed
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse
echo "=== validating 400 DESIGNED layouts (32 workers, 3 seeds) ==="
python -m my_methods.bayesian.validate_designed 32 3 400
