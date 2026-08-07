#!/bin/bash -l
# ===========================================================================
# SP, ZSC-EVAL PARITY.  Core 7 layouts x seeds 5..15  =  77 runs.
# ===========================================================================
# This replaces the seed-1-only run documented in SP/CARC_RUNS.md. It does NOT
# overwrite it: everything here lands under steakhouse_zsc/, the old runs stay
# at steakhouse_sp/ exactly where filter/baseline.py looks for them.
#
# WHY 11 SEEDS.  ZSC-Eval's train_sp.sh loops seed 5..15 for every layout.
# Eleven independent runs, never merged -- what they buy is a distribution
# instead of an anecdote. The existing table is n=1, so a 0.00 there is not
# evidence a layout is unlearnable and a 60.00 there has no error bar.
#
# WHY THESE 7 LAYOUTS.  All solved at seed 1, all small (25-81 cells), and all
# clustered/contention -- which LAYOUTS.md ranks highest for robot influence.
# The layouts self-play can solve and the layouts the FOV result needs are the
# same layouts, so the narrow set costs nothing scientifically and buys enough
# compute to afford full ZSC seed counts across three algorithms.
#
# WHY IT REPLACES A RUN THAT ALREADY WORKED.  Two reasons, both structural:
#   1. self_play.py saved every checkpoint to ONE filename, so a finished run
#      left only final weights. FCP's population is init/mid/final per seed --
#      it is not buildable from what exists. self_play_pop.py writes
#      actor_periodic_<ep>.pt, a new file each save.
#   2. There was no per-episode score log, so ZSC's rule for picking the "mid"
#      checkpoint (the one scoring half the final score) could not be applied
#      after the fact. progress.jsonl fixes that.
#
# ARRAY INDEX -> (layout, seed):  layout = id / 11,  seed = 5 + id % 11
# so tasks 0-10 are gc00 seeds 5-15, tasks 11-21 are gc03 seeds 5-15, and so on.

#SBATCH --job-name=zsc_sp
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-76%24
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse

LAYOUTS=(steak_gc00 steak_gc03 steak_gc04 steak_gc06 steak_cram steak_cram2 steak_gs00)
N_SEEDS=11
SEED_BEGIN=5

LAYOUT="${LAYOUTS[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}"
SEED=$((SEED_BEGIN + SLURM_ARRAY_TASK_ID % N_SEEDS))

SAVE_DIR="/scratch1/$USER/steakhouse_zsc/sp/${LAYOUT}_seed${SEED}"
mkdir -p "$SAVE_DIR"

# ENTROPY ANNEAL -- ZSC-Eval's two-segment schedule (train_sp.sh):
#     0.2 -> 0.05 over the first 5e6 steps, then 0.05 -> 0.01 over the next 5e6
# The validated seed-1 runs used a single line 0.2 -> 0.01 over 1e7, which sits
# at 0.105 at the midpoint versus 0.05 here -- about twice the exploration
# pressure through the middle of training. All three arms in this wave (SP, E3T,
# SP-eps) share these exact numbers so a gap between them is attributable to the
# algorithm and not to the schedule.
# To reproduce the old behaviour:  ENT_ARGS="--entropy_coefs 0.2 0.01 --entropy_coef_horizons 0 1e7"
if [ -z "${ENT_ARGS+x}" ]; then
  ENT_ARGS="--entropy_coefs 0.2 0.05 0.01 --entropy_coef_horizons 0 5e6 1e7"
fi

echo "######## STAGING LAYOUTS ########"
# from_layout_name() only reads overcooked_ai_py/data/layouts. cp -n so
# concurrent array tasks cannot clobber each other.
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/ 2>/dev/null || true

echo "######## ENV ########"
python -c "import torch,numpy,sys; print('python',sys.version.split()[0]); print('torch',torch.__version__,'| cuda:',torch.cuda.is_available()); print('numpy',numpy.__version__)"
echo "task=$SLURM_ARRAY_TASK_ID layout=$LAYOUT seed=$SEED save_dir=$SAVE_DIR"
echo "ent_args=[$ENT_ARGS]"

echo "######## SMOKE TEST (must pass before the long run) ########"
# writes into smoke/ so its junk checkpoints can never be confused with the
# real ones -- the exact trap CARC_RUNS.md section 5 documents
python -m fov.robot.policy.new.official_baselines.SP.self_play_pop \
    --layouts "$LAYOUT" --run_name "$LAYOUT" --seed "$SEED" \
    --n_rollout_threads 4 --episode_length 50 --horizon 50 \
    --num_env_steps 800 --ppo_epoch 2 --num_mini_batch 1 \
    --save_interval 1 --log_interval 1 \
    --save_dir "$SAVE_DIR/smoke"

echo "######## TRAIN ########"
python -m fov.robot.policy.new.official_baselines.SP.self_play_pop \
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
    --use_valuenorm \
    $ENT_ARGS \
    --reward_shaping_horizon 1e8 \
    --save_interval 25 \
    --log_interval 10 \
    --save_dir "$SAVE_DIR"

echo "######## DONE ########"
ls -la "$SAVE_DIR"
wc -l "$SAVE_DIR/progress.jsonl"
