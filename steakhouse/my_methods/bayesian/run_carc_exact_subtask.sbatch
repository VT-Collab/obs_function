#!/bin/bash -l
# MISHA NEW CHANGE - additive file, does not modify run_carc.sbatch.
#SBATCH --job-name=steak_island_exact_subtask
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# fail loudly instead of silently continuing past a broken step (this is what
# let the earlier job run build/compare against a half-installed environment
# after `pip install` failed on a Mac-only package - see CARC_NOTES.md)
set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

echo "=== environment check ==="
which python
python --version
which pip

cd ~/steakhouse

echo "=== installing dependencies ==="
# using a minimal, known-mutually-compatible package list instead of the full
# 171-package local freeze - that full list has internal version conflicts
# (tensorflow's old typing_extensions pin vs newer packages elsewhere) that
# don't matter for this experiment since it's pure numpy/torch/search, no
# tensorflow/keras involved at all. These 6 exact versions are the ones that
# were actually resolved and running together on the local machine.
pip install -r requirements-minimal.txt

echo "=== building QMDP planner (skips if already cached) ==="
python -m my_methods.bayesian.build_qmdp_steak_island

echo "=== running static vs dynamic-Bayesian comparison, exact subtask ==="
python -m my_methods.bayesian.qmdp_bayesian_vs_static_exact_subtask
