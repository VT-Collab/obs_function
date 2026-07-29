#!/bin/bash -l
# MISHA NEW CHANGE - v2: room sizes capped to 15-17x9-11 (larger rooms were
# tanking throughput), num_items_for_steak=2 removed (suspected IndexError
# source), layout names now include SLURM_JOB_ID to avoid cache collisions
# with the still-running first batch (job 10470079).
#SBATCH --job-name=fov_parallel_search_v2
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

cd ~/steakhouse

echo "=== running large parallel FOV divergence search v2 (faster room sizes) ==="
python -m my_methods.bayesian.fov_parallel_layout_search 128 32
