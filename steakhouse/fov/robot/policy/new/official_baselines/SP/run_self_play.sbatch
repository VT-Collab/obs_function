#!/bin/bash -l
# Self-play PPO baseline for steakhouse. Both chefs driven by one network.
#
# ONE policy trained ACROSS many layouts. Every grid is padded onto a shared
# canvas as big as the biggest layout in the pool (same approach as ZSC-Eval's
# layout_generator.mdp_gen_fn_from_dict / embed_grid), with a random offset
# each reset so the policy cannot memorize absolute coordinates.
#
# CPU partition on purpose. The bottleneck for overcooked-scale MAPPO is
# envs.step(), which is pure python and CPU-bound and cannot be batched onto a
# GPU -- the conv stack is microseconds either way. More parallel envs is the
# real throughput lever, so cpus-per-task is the knob, not --gres=gpu.
#
# submit:  sbatch fov/robot/policy/new/official_baselines/SP/run_self_play.sbatch
# pick a pool: sbatch --export=ALL,POOL=contention ... run_self_play.sbatch
#   POOL=tier1       25 layouts, INFL-ranked, canvas 15x11   (default)
#   POOL=contention   6 highest-influence cramped layouts, canvas 15x10
#   POOL=tight        5 cramped layouts only, canvas 13x9 -- drops mid_1,
#                     which is (15,10) and single-handedly sets the canvas
#                     for the contention pool. much less padding waste.
#   or POOL=<comma,separated,names>

#SBATCH --job-name=sp_steak
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

# ---------------------------------------------------------------- layouts
# Tier 1 of fov/layouts_final/LAYOUTS.md: PASS all five gates AND influential
# (INFL >= 8, i.e. seeing the robot measurably changes the human). Ranked by
# INFL, highest first. steak_gs01 is excluded -- it fails team-win at fov30.
TIER1="steak_gc00,steak_gc06,steak_gc04,steak_cram2,steak_cram,steak_gs00,steak_mid_1,steak_gc03,steak_gc05,steak_side_3,steak_gs07,steak_side_4,steak_api,steak_gc07,steak_gc01,steak_gs03,steak_gs04,steak_parrallel,steak_gs02,steak_gs08,steak_mid_2,steak_gs09,steak_none_3,steak_tshape,steak_gs05"

# The contention layouts -- LAYOUTS.md headline finding is that cramped /
# clustered kitchens drive robot influence (+12.5% team throughput aware vs
# blind, vs +4.7% for open rooms). They also pad to a much smaller canvas.
CONTENTION="steak_gc00,steak_gc06,steak_gc04,steak_cram2,steak_cram,steak_mid_1"

# same idea minus steak_mid_1, which is (15,10) and on its own drags the
# canvas from 13x9 up to 15x10 -- nearly doubling the dead space every one of
# the small gc* kitchens has to be padded into.
TIGHT="steak_gc00,steak_gc06,steak_gc04,steak_cram2,steak_cram"

POOL="${POOL:-tier1}"
case "$POOL" in
    tier1)      LAYOUTS="$TIER1" ;;
    contention) LAYOUTS="$CONTENTION" ;;
    tight)      LAYOUTS="$TIGHT" ;;
    *)          LAYOUTS="$POOL" ;;
esac

SEED="${SEED:-1}"
RUN_NAME="${RUN_NAME:-$POOL}"
MODE="generalist"

# ------------------------------------------------------------ ARRAY MODE
# Submitted with --array=0-24? Then this is a SPECIALIST run: the array index
# picks ONE layout out of the pool and trains a policy on only that kitchen.
# No padding, no dead space -- this is what ZSC-Eval actually does for
# overcooked (train_sp.sh takes a single layout=$1; its padding code is only
# ever called from tests). Gives the per-layout column LAYOUTS.md expects.
#
#   sbatch --array=0-24%10 --export=ALL,POOL=tier1 run_self_play.sbatch
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    IFS=',' read -ra _POOL_ARR <<< "$LAYOUTS"
    LAYOUTS="${_POOL_ARR[$SLURM_ARRAY_TASK_ID]}"
    RUN_NAME="$LAYOUTS"
    MODE="specialist"
    echo "array task $SLURM_ARRAY_TASK_ID / ${#_POOL_ARR[@]} -> $LAYOUTS"
fi

# home1 quota is tight -- checkpoints go to scratch, not home
SAVE_DIR="${SAVE_DIR:-/scratch1/$USER/steakhouse_sp/${MODE}/${RUN_NAME}_seed${SEED}}"
mkdir -p "$SAVE_DIR"

# ------------------------------------------------- stage the layout library
# from_layout_name() only reads overcooked_ai_py/data/layouts, so the
# validated library has to be copied in before anything can load it.
echo "######## STAGING LAYOUTS ########"
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/ 2>/dev/null || true
echo "staged: $(ls overcooked_ai_py/data/layouts/*.layout | wc -l) layout files total"

echo "######## ENV ########"
python - <<'EOF'
import torch, numpy, sys
print("python ", sys.version.split()[0])
print("torch  ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy  ", numpy.__version__)
EOF
echo "mode=$MODE pool=$POOL seed=$SEED run=$RUN_NAME cpus=$SLURM_CPUS_PER_TASK"
echo "layouts=$LAYOUTS"
echo "save_dir=$SAVE_DIR"
# NOTE VecSteakEnv steps its envs SEQUENTIALLY -- extra cpus only help the
# torch conv/linear threads, not the env loop. Asking for 32 cpus does not
# make the rollout 32x faster. If you want real env parallelism, that is a
# SubprocVecEnv change in utils/env_wrapper.py, not a slurm flag.

echo "######## SMOKE TEST (must pass before the long run) ########"
python -m fov.robot.policy.new.official_baselines.SP.self_play \
    --layouts "$LAYOUTS" \
    --run_name "$RUN_NAME" \
    --n_rollout_threads 4 \
    --episode_length 40 \
    --horizon 40 \
    --num_env_steps 480 \
    --ppo_epoch 2 \
    --num_mini_batch 2 \
    --data_chunk_length 10 \
    --save_dir "$SAVE_DIR/smoke"

echo "######## TRAIN ########"
python -m fov.robot.policy.new.official_baselines.SP.self_play \
    --layouts "$LAYOUTS" \
    --run_name "$RUN_NAME" \
    --seed "$SEED" \
    --n_rollout_threads 32 \
    --episode_length 200 \
    --horizon 260 \
    --n_orders 4 \
    --num_env_steps 40000000 \
    --hidden_size 64 \
    --recurrent_N 1 \
    --lr 7e-4 \
    --critic_lr 1e-3 \
    --ppo_epoch 10 \
    --num_mini_batch 2 \
    --data_chunk_length 10 \
    --entropy_coef 0.01 \
    --shaping_anneal_eps 500 \
    --use_linear_lr_decay \
    --save_interval 50 \
    --log_interval 10 \
    --save_dir "$SAVE_DIR"

echo "######## DONE ########"
ls -la "$SAVE_DIR"
