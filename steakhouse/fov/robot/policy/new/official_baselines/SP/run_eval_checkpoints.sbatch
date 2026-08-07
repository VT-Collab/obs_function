#!/bin/bash -l
# Evaluate trained specialist checkpoints by actually PLAYING them.
#
# MUST run on a compute node. numpy segfaults on the CARC login node:
#   "Importing the numpy C-extensions failed ... Segmentation fault (core dumped)"
#
#   sbatch fov/robot/policy/new/official_baselines/SP/run_eval_checkpoints.sbatch
#   LAYOUTS=steak_gc00,steak_api EPISODES=10 sbatch .../run_eval_checkpoints.sbatch
#   MODE=stochastic sbatch .../run_eval_checkpoints.sbatch

#SBATCH --job-name=ckeval
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse

# all 25 Tier 1 layouts by default, INFL-ranked order
LAYOUTS="${LAYOUTS:-steak_gc00,steak_gc06,steak_gc04,steak_cram2,steak_cram,steak_gs00,steak_mid_1,steak_gc03,steak_gc05,steak_side_3,steak_gs07,steak_side_4,steak_api,steak_gc07,steak_gc01,steak_gs03,steak_gs04,steak_parrallel,steak_gs02,steak_gs08,steak_mid_2,steak_gs09,steak_none_3,steak_tshape,steak_gs05}"
EPISODES="${EPISODES:-5}"
MODE="${MODE:-deterministic}"

python fov/robot/policy/new/official_baselines/SP/eval_checkpoints.py \
    "$LAYOUTS" "$EPISODES" "$MODE"
