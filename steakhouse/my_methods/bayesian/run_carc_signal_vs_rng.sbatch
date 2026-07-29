#!/bin/bash -l
# MISHA NEW CHANGE - does the recorded divergence of the 23 curated layouts
# survive when the RNG is held fixed? Splits each layout's divergence into the
# FOV effect (across FOVs at a fixed seed) vs the RNG noise floor (across seeds
# at a fixed FOV). See fov_signal_vs_rng.py for why this matters.
#SBATCH --job-name=fov_signal_vs_rng
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=90G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

cd ~/steakhouse

echo "=== FOV signal vs RNG noise across all 23 curated layouts ==="
python -m my_methods.bayesian.fov_signal_vs_rng 23 3 120
