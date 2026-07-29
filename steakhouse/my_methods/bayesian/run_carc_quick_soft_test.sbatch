#!/bin/bash -l
#SBATCH --job-name=fov_quick_soft_test
#SBATCH --account=biyik_1173
#SBATCH --partition=debug
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse
python -m my_methods.bayesian.quick_soft_test
