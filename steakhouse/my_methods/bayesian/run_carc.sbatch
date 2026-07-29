#!/bin/bash -l
#SBATCH --job-name=steak_island_qmdp
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# SLURM batch scripts run as non-login shells by default, so `module` (defined
# in the login profile) isn't available unless we force a login shell via the
# "-l" shebang flag above - without it, `module load` silently fails and
# everything below runs against the wrong (system) python.
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai

echo "=== environment check ==="
which python
python --version
which pip

cd ~/steakhouse

echo "=== installing dependencies ==="
pip install -r steakhouse-ai-requirements.txt

echo "=== building QMDP planner (skips if already cached) ==="
python -m my_methods.bayesian.build_qmdp_steak_island

echo "=== running static vs dynamic-Bayesian comparison ==="
python -m my_methods.bayesian.qmdp_bayesian_vs_static
