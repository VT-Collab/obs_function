#!/bin/bash -l
# ===========================================================================
# RUN 1 of 2 -- GENERALIST.  ONE policy across ALL 25 Tier 1 layouts.
# ===========================================================================
# Submitted as job 10789482 via:
#     sbatch --export=ALL,POOL=tier1,SEED=1 run_self_play.sbatch
# This file is that exact run, unrolled into a standalone script.
#
# Every kitchen is padded onto a shared 15x11 canvas (as big as the biggest
# layout in the pool) and dropped in at a RANDOM OFFSET each reset, so the
# policy cannot memorize absolute coordinates. Padded cells are written as
# 'X' counter terrain, not zeros -- all-zero terrain channels mean "floor"
# in features.py, which would have told the agent the dead space outside the
# kitchen was walkable.
#
# THE COST: canvas fill fraction per layout, measured at startup --
#     steak_gc00  0.15   <- your HIGHEST-influence layout (INFL 28.5).
#     steak_gs00  0.18      85% of its conv work is dead space.
#     steak_gc01  0.18
#     steak_gc06  0.21
#     ...
#     steak_mid_2 1.00   <- the only one that fills the canvas
#
# That waste is the price of one policy that works in any kitchen.
# Compare against run_specialists.sbatch, which pays no padding at all.
#
# NOTE this is NOT what ZSC-Eval does. Their padding code (layout_generator.
# mdp_gen_fn_from_dict / embed_grid) is only ever called from overcooked_test.py
# and benchmarking.py -- never from a training script. train_sp.sh takes
# `layout=$1`, one layout per run. They never needed padding because their
# overcooked layouts are all 5x4 / 5x5 / 9x5. Your Tier 1 spans (5,5)-(15,11).

#SBATCH --job-name=sp_generalist
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse

# Tier 1 of fov/layouts_final/LAYOUTS.md: passes all five gates AND influential
# (INFL >= 8). Ranked by INFL, highest first. steak_gs01 excluded -- it fails
# team-win at fov30.
LAYOUTS="steak_gc00,steak_gc06,steak_gc04,steak_cram2,steak_cram,steak_gs00,steak_mid_1,steak_gc03,steak_gc05,steak_side_3,steak_gs07,steak_side_4,steak_api,steak_gc07,steak_gc01,steak_gs03,steak_gs04,steak_parrallel,steak_gs02,steak_gs08,steak_mid_2,steak_gs09,steak_none_3,steak_tshape,steak_gs05"

SEED="${SEED:-1}"
SAVE_DIR="/scratch1/$USER/steakhouse_sp/generalist/tier1_seed${SEED}"
mkdir -p "$SAVE_DIR"

# from_layout_name() only reads overcooked_ai_py/data/layouts, so the validated
# library has to be copied in before anything can load it.
echo "######## STAGING LAYOUTS ########"
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/ 2>/dev/null || true
echo "staged: $(ls overcooked_ai_py/data/layouts/*.layout | wc -l) layout files"

echo "######## ENV ########"
python -c "import torch,numpy,sys; print('python',sys.version.split()[0]); print('torch',torch.__version__,'| cuda:',torch.cuda.is_available()); print('numpy',numpy.__version__)"
echo "mode=generalist seed=$SEED cpus=$SLURM_CPUS_PER_TASK save_dir=$SAVE_DIR"
# NOTE VecSteakEnv steps its envs SEQUENTIALLY. Extra cpus only help the torch
# conv/linear threads, not the env loop -- 32 cpus does NOT make the rollout
# 32x faster. Real env parallelism would be a SubprocVecEnv change in
# utils/env_wrapper.py, not a slurm flag.

echo "######## SMOKE TEST (must pass before the long run) ########"
python -m fov.robot.policy.new.official_baselines.SP.self_play \
    --layouts "$LAYOUTS" --run_name tier1 \
    --n_rollout_threads 4 --episode_length 50 --horizon 50 \
    --num_env_steps 800 --ppo_epoch 2 --num_mini_batch 1 \
    --save_dir "$SAVE_DIR/smoke"

echo "######## TRAIN ########"
python -m fov.robot.policy.new.official_baselines.SP.self_play \
    --layouts "$LAYOUTS" \
    --run_name tier1 \
    --seed "$SEED" \
    --n_rollout_threads 50 \
    --episode_length 400 \
    --horizon 400 \
    --n_orders 4 \
    --num_env_steps 10000000 \
    --hidden_size 64 \
    --recurrent_N 1 \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --ppo_epoch 15 \
    --num_mini_batch 1 \
    --data_chunk_length 10 \
    --value_loss_coef 1.0 \
    --use_huber_loss \
    --entropy_coef_start 0.2 \
    --entropy_coef 0.01 \
    --entropy_coef_horizon 1e7 \
    --reward_shaping_horizon 1e8 \
    --save_interval 25 \
    --log_interval 10 \
    --save_dir "$SAVE_DIR"

echo "######## DONE ########"
ls -la "$SAVE_DIR"
