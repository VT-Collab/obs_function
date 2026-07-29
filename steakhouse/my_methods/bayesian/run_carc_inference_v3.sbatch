#!/bin/bash -l
# MISHA NEW CHANGE - Bayesian FOV inference across the curated v3 layout set.
# These are the layouts that carry REAL (RNG-free) FOV-driven subtask divergence,
# each above a 1.5x signal-to-noise margin. Reads kb_update_delay per layout from
# its header and applies it to both ground truth and the shadow hypotheses.
#SBATCH --job-name=fov_inf_v3
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse

# both likelihood modes, 3 seeds: 17 layouts x 3 FOVs x 2 modes x 3 seeds = 306 trials
echo "=== FOV inference over curated v3 layouts (both modes, 3 seeds) ==="
python -m my_methods.bayesian.fov_inference_v3 32 all 3
