#!/bin/bash -l
# ===========================================================================
# RUN 2 of 2 -- SPECIALISTS.  25 separate policies, ONE per layout.
# ===========================================================================
# Submitted as job array 10789524 via:
#     sbatch --array=0-24%10 --cpus-per-task=8 --mem=32G \
#            --export=ALL,POOL=tier1,SEED=1 run_self_play.sbatch
# This file is that exact run, unrolled into a standalone script.
#
# THIS IS WHAT ZSC-EVAL ACTUALLY DOES. Their train_sp.sh takes `layout=$1`
# -- one layout per run, one policy per kitchen. Their padding code
# (layout_generator.mdp_gen_fn_from_dict / embed_grid) is only ever called
# from overcooked_test.py and benchmarking.py, never from training.
#
# WHY IT MATTERS HERE: no padding, so no dead space. steak_gc00 is (5,5) and
# trains as (5,5) -- versus 15% canvas fill in the generalist run. The CNN's
# Linear(32*W*H, 64) is sized per layout, which is exactly why these
# checkpoints are NOT interchangeable across layouts. That is the tradeoff:
# 25 policies that each work in one kitchen, vs 1 policy that works in all of
# them but wastes most of its compute on padding.
#
# Also gives the per-layout column that fov/layouts_final/LAYOUTS.md is
# already organized around, so the baseline numbers drop straight into the
# INFL table.
#
# %10 caps concurrency at 10 running tasks so this does not eat the queue.
# 8 cpus rather than 32: VecSteakEnv steps SEQUENTIALLY, so extra cores only
# feed the torch conv threads. More concurrent jobs beats more cores per job.

#SBATCH --job-name=sp_specialist
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-24%10
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse

# Same Tier 1 pool as the generalist run, but the array index picks ONE.
# Order is INFL-ranked, highest first -- so array task 0 is steak_gc00
# (INFL 28.5, the most influential layout in the library) and task 24 is
# steak_gs05 (INFL 8.5, the Tier 1 floor).
TIER1="steak_gc00,steak_gc06,steak_gc04,steak_cram2,steak_cram,steak_gs00,steak_mid_1,steak_gc03,steak_gc05,steak_side_3,steak_gs07,steak_side_4,steak_api,steak_gc07,steak_gc01,steak_gs03,steak_gs04,steak_parrallel,steak_gs02,steak_gs08,steak_mid_2,steak_gs09,steak_none_3,steak_tshape,steak_gs05"

IFS=',' read -ra POOL_ARR <<< "$TIER1"
LAYOUT="${POOL_ARR[$SLURM_ARRAY_TASK_ID]}"

SEED="${SEED:-1}"
SAVE_DIR="/scratch1/$USER/steakhouse_sp/specialist/${LAYOUT}_seed${SEED}"
mkdir -p "$SAVE_DIR"

# from_layout_name() only reads overcooked_ai_py/data/layouts. cp -n so the 25
# array tasks staging concurrently do not clobber each other.
echo "######## STAGING LAYOUTS ########"
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/ 2>/dev/null || true

echo "######## ENV ########"
python -c "import torch,numpy,sys; print('python',sys.version.split()[0]); print('torch',torch.__version__,'| cuda:',torch.cuda.is_available()); print('numpy',numpy.__version__)"
echo "mode=specialist array_task=$SLURM_ARRAY_TASK_ID/${#POOL_ARR[@]} layout=$LAYOUT"
echo "seed=$SEED cpus=$SLURM_CPUS_PER_TASK save_dir=$SAVE_DIR"

echo "######## SMOKE TEST (must pass before the long run) ########"
python -m fov.robot.policy.new.official_baselines.SP.self_play \
    --layouts "$LAYOUT" --run_name "$LAYOUT" \
    --n_rollout_threads 4 --episode_length 50 --horizon 50 \
    --num_env_steps 800 --ppo_epoch 2 --num_mini_batch 1 \
    --save_dir "$SAVE_DIR/smoke"

echo "######## TRAIN ########"
python -m fov.robot.policy.new.official_baselines.SP.self_play \
    --layouts "$LAYOUT" \
    --run_name "$LAYOUT" \
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
