#!/bin/bash -l
# ===========================================================================
# E3T.  Core 7 layouts x seeds 1..5  =  35 runs.
# ===========================================================================
# ZSC-Eval's train_e3t.sh: seeds 1..5, 1e7 env steps, epsilon 0.25,
# weights_copy_factor 0.1. Single stage -- no population, no pool, nothing to
# pre-train, which is why this can launch at the same time as SP instead of
# after it.
#
# WHAT E3T IS DOING: the partner is a slow copy of the ego (tau=0.1 per
# episode) that also takes a uniformly random action 25% of the time. Self-play
# fails at zero-shot coordination because it trains against a perfectly
# predictable partner and converges on a private convention; E3T makes the
# partner both lagged and noisy so no such convention survives.
#
# THAT NOISE IS THE POINT FOR US. A limited-FOV human constantly does things
# the robot could not have predicted -- it walks to a station that is already
# occupied because it cannot see the occupant. epsilon-noise is the cheapest
# possible stand-in for that, and it is injected with zero FOV-human exposure.
#
# WHY 100 THREADS AND NOT 50.  ZSC-Eval uses 50 for SP and 100 for E3T, and
# that is not arbitrary: SP trains on BOTH chefs, so 50 kitchens give 100 rows
# per batch. Here only the ego learns, so 50 kitchens would give 50 rows -- half
# the batch SP gets, on the same 1e7 env steps. 100 kitchens restores parity.
# Total env steps are unchanged, so wall-clock is unchanged; what changes is 250
# PPO updates on double-width batches instead of 500 on single-width ones,
# exactly as in train_e3t.sh.
#
# ARRAY INDEX -> (layout, seed):  layout = id / 5,  seed = 1 + id % 5
#
# Runs on the SECOND account so it does not compete with the SP array for the
# same allocation -- SP is the long pole (FCP stage 1 waits on it).

#SBATCH --job-name=zsc_e3t
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-34%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e

module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse

LAYOUTS=(steak_gc00 steak_gc03 steak_gc04 steak_gc06 steak_cram steak_cram2 steak_gs00)
N_SEEDS=5
SEED_BEGIN=1

LAYOUT="${LAYOUTS[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}"
SEED=$((SEED_BEGIN + SLURM_ARRAY_TASK_ID % N_SEEDS))

# tau = 0.1 is E3T. Override to 1.0 for the SP-epsilon ablation -- see
# run_sp_eps.sbatch, which is this file with two variables changed.
TAU="${TAU:-0.1}"
EPS="${EPS:-0.25}"
TAG="${TAG:-e3t}"

SAVE_DIR="/scratch1/$USER/steakhouse_zsc/${TAG}/${LAYOUT}_seed${SEED}"
mkdir -p "$SAVE_DIR"

# ENTROPY ANNEAL -- ZSC-Eval's two-segment schedule, knee at 5e6.
# train_e3t.sh actually puts E3T's knee at 6e6 rather than SP's 5e6. We use 5e6
# for all three arms in this wave on purpose: an SP-vs-E3T gap has to be
# attributable to the algorithm, and a 1e6-step difference in the knee (0.05 vs
# 0.075 at the 5e6 mark) is not worth confounding the comparison over.
if [ -z "${ENT_ARGS+x}" ]; then
  ENT_ARGS="--entropy_coefs 0.2 0.05 0.01 --entropy_coef_horizons 0 5e6 1e7"
fi

echo "######## STAGING LAYOUTS ########"
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/ 2>/dev/null || true

echo "######## ENV ########"
python -c "import torch,numpy,sys; print('python',sys.version.split()[0]); print('torch',torch.__version__,'| cuda:',torch.cuda.is_available()); print('numpy',numpy.__version__)"
echo "task=$SLURM_ARRAY_TASK_ID layout=$LAYOUT seed=$SEED tag=$TAG tau=$TAU eps=$EPS"
echo "save_dir=$SAVE_DIR  ent_args=[$ENT_ARGS]"

echo "######## SMOKE TEST (must pass before the long run) ########"
python -m fov.robot.policy.new.official_baselines.E3T.e3t \
    --layouts "$LAYOUT" --run_name "$LAYOUT" --seed "$SEED" \
    --n_rollout_threads 4 --episode_length 50 --horizon 50 \
    --num_env_steps 800 --ppo_epoch 2 --num_mini_batch 1 \
    --epsilon "$EPS" --weights_copy_factor "$TAU" \
    --save_interval 1 --log_interval 1 \
    --save_dir "$SAVE_DIR/smoke"

echo "######## TRAIN ########"
python -m fov.robot.policy.new.official_baselines.E3T.e3t \
    --layouts "$LAYOUT" \
    --run_name "$LAYOUT" \
    --seed "$SEED" \
    --n_rollout_threads 100 \
    --episode_length 400 \
    --horizon 400 \
    --n_orders 4 \
    --num_env_steps 10000000 \
    --hidden_size 64 \
    --recurrent_N 1 \
    --epsilon "$EPS" \
    --weights_copy_factor "$TAU" \
    --random_index \
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
