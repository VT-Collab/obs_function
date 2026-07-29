#!/bin/bash -l
# MISHA NEW CHANGE - large parallel random search for layouts with genuine
# 3-way subtask-level FOV divergence, sustained across the full episode.
#SBATCH --job-name=fov_parallel_search
#SBATCH --account=biyik_1173
#SBATCH --partition=debug
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

cd ~/steakhouse

echo "=== running large parallel FOV divergence search (debug partition, was idle) ==="
python -m my_methods.bayesian.fov_parallel_layout_search 96 24
