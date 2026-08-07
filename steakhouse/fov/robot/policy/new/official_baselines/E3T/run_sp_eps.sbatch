#!/bin/bash -l
# ===========================================================================
# SP-EPSILON.  The E3T ablation.  Core 7 layouts x seeds 1..5  =  35 runs.
# ===========================================================================
# Same script, same code path, ONE number different:
#
#     E3T        --weights_copy_factor 0.1    partner is a LAGGED copy of ego
#     SP-eps     --weights_copy_factor 1.0    partner IS the ego, exactly
#
# At tau = 1.0 the soft copy theta_p <- (1-tau)*theta_p + tau*theta_ego reduces
# to theta_p = theta_ego, so the partner has no lag at all and the only thing
# separating this from plain self-play is the 25% random action.
#
# WHAT IT ANSWERS: E3T changes two things at once. If E3T beats SP against the
# FOV human, this run says whether that came from the lagged partner or purely
# from partner noise. If SP-eps captures most of the gain, then the useful
# ingredient is "your partner sometimes does something unexpected" -- which is
# a much more direct statement about limited-FOV partners than E3T's full
# machinery, and a much simpler thing to build on.
#
# It costs one flag and no new code, which is the only reason it is in wave 1.
# Sharing e3t.py rather than forking it also guarantees the two arms cannot
# differ anywhere except tau.

#SBATCH --job-name=zsc_speps
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

TAU=1.0            # <- the ablation. exact copy, no lag.
EPS="${EPS:-0.25}"
TAG=sp_eps

SAVE_DIR="/scratch1/$USER/steakhouse_zsc/${TAG}/${LAYOUT}_seed${SEED}"
mkdir -p "$SAVE_DIR"

# identical to run_e3t.sbatch -- the ablation must differ in tau and nothing else
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
