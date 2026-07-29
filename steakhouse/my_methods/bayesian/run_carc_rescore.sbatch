#!/bin/bash -l
# MISHA NEW CHANGE - re-score fov/layouts_v3 with the PHASE-CORRECTED divergence
# metric (run-length alignment, substitutions only) and the stricter noise gate
# (compare min-signal against MAX seed-pair noise, not the mean). Reports which
# layouts still qualify, plus liveness (DONE vs STUCK) which was never checked.
#SBATCH --job-name=fov_rescore
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=90G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse
echo "=== re-scoring v3 layouts with corrected metric ==="
python -m my_methods.bayesian.rescore_v3_layouts 17 3
