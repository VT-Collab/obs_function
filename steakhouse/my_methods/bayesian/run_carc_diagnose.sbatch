#!/bin/bash -l
# MISHA NEW CHANGE - quick diagnostic job, additive, does not modify other sbatch files.
#SBATCH --job-name=steak_island_diagnose
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

cd ~/steakhouse
python -m my_methods.bayesian.diagnose_exact_subtask
