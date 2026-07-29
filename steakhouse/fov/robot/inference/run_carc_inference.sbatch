#!/bin/bash -l
# MISHA NEW CHANGE - Bayesian FOV inference accuracy on the validated layouts,
# using the from-scratch limited-vision human. One shadow agent per candidate
# FOV, each maintaining its own FOV-gated decaying beliefs; log-space posterior.
#
# These layouts/triples come from CARC job 10487033, which established that all
# 12 tested layouts show substantial FOV-driven subtask divergence and that
# wider FOV delivers more. That divergence is what the likelihood reads.
#SBATCH --job-name=fov_inference
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse
# 7 layouts x 3 FOVs x 10 seeds = 210 trials
echo "=== Bayesian FOV inference on validated layouts (10 seeds) ==="
python -m fov.robot.inference.evaluate_inference 24 10
