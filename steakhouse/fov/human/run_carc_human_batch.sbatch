#!/bin/bash -l
# MISHA NEW CHANGE - thorough validation of the FROM-SCRATCH limited-vision human
# across every available steak layout x 6 FOVs x 3 seeds, scoring BOTH required
# conditions: substantial subtask divergence, and better performance with wider
# FOV. Verified locally on steak_island first (all 6 pairs diverge 3-17;
# delivered 1.0/1.0/2.0 at fov 30/90/360 with 64/15.5/3 look-steps) - this
# establishes how broadly that holds.
#SBATCH --job-name=fov_human_batch
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse
echo "=== from-scratch human: layouts x 6 FOVs x 3 seeds ==="
python -m fov.human.batch_validate 24 3
