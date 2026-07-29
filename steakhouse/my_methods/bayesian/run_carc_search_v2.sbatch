#!/bin/bash -l
# MISHA NEW CHANGE - v2 layout/settings search for GENUINE FOV-driven subtask
# divergence: RNG reset to the same seed across FOVs (so the score isolates the
# FOV effect instead of measuring auto_unstuck's dice), replicated across seeds,
# with a teammate that actually works the task so the world keeps changing.
# See fov_divergence_search_v2.py for the full rationale.
#
# Each trial builds its own MediumLevelPlanner (~3-7 min, unavoidable per unique
# layout) and then runs n_seeds x 3 FOVs rollouts, so it is ~3x v1's per-trial
# sim cost on top of the build. Builds dominate; wall-clock stays close to
# (n_trials / n_workers) * build_time.
#SBATCH --job-name=fov_search_v3
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

cd ~/steakhouse

# The watched human is the PLAIN SteakLimitVisionHumanModel. Measured
# head-to-head, the sticky wrapper stalls the human ~20 steps in with only
# 1-2 subtasks touched, while the plain model runs the full steak workflow
# (6+ subtasks, up to 177 steps) AND shows real FOV-driven divergence at a
# fixed seed. See fov_divergence_search_v2.py's HUMAN_CLS comment.
export FOV_HUMAN=plain

# Wall-embedded v3 layouts build in 27-62s (vs 3-7 min for v1's free-standing
# rooms), so a much larger sweep fits in the same wall-clock. Each trial's
# planner cache (120-220MB) is deleted right after use - keeping 384 of them
# would be ~60GB against only ~25GB free on /home1.
# n_trials n_workers n_seeds
echo "=== v4 search: 768 layouts, 32 workers, 3 seeds ==="
echo "=== gate: real subtask divergence (order/choice, NOT timing) ==="
echo "=== plus FULL-FOV CONTROL: 180deg human must finish, <5% stalls ==="
# seed offset 50000: the previous run defaulted to 7000 and re-explored
# layouts an earlier local batch had already found (4 exact duplicates).
python -m my_methods.bayesian.fov_divergence_search_v2 768 32 3 50000
