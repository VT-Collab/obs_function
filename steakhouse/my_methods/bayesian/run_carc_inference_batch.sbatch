#!/bin/bash -l
# MISHA NEW CHANGE - Bayesian FOV inference accuracy across all 23 validated
# sustained-divergence layouts, reusing already-built MediumLevelPlanner caches
# (no rebuild). Sweeps BOTH likelihood modes (greedy = minigrid-faithful
# epsilon-greedy, boltzmann = softmax over one-step-ahead motion costs), so the
# comparison is measured rather than assumed: 23 layouts x 3 FOVs x 2 modes
# = 138 trials.
#
# Moved off the debug partition (55min cap) - at ~10-12 min/trial and 138
# trials, even 32 workers needs a couple of hours.
#SBATCH --job-name=fov_inference_batch
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

cd ~/steakhouse

# 120 steps to match the horizon the layouts were VALIDATED at
# (fov_search_results.md measured late-half divergence over steps 60-120; running
# the filter for only 60 steps would look at the wrong window entirely).
export FOV_N_STEPS=120

# greedy only for this pass. The question this run has to answer first is
# "do the FOV hypotheses diverge AT ALL when they spectate one shared
# trajectory?" - and goal/subtask selection is deterministic, so the answer is
# identical under both likelihood modes. Running one mode halves the cost;
# sweep both once divergence is confirmed to exist.
echo "=== Bayesian FOV inference batch: 23 layouts x 3 FOVs, greedy, 120 steps ==="
echo "=== filter: my_methods/bayesian/fov_bayes_filter.py (log-space, minigrid style) ==="
echo "=== KEY METRIC THIS PASS: div= (goal divergence rate). If 0 everywhere, ==="
echo "=== the spectator setup carries no FOV signal and the design needs to change. ==="
python -m my_methods.bayesian.fov_inference_batch 32 greedy 23
