#!/bin/bash
#SBATCH --job-name=filter_eval
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home1/%u/steakhouse/logs/filter_%A_%a.out
# =============================================================================
# baseline vs baseline+FOV-module, across ZSC wave-1 policies.
#
#   ARRAY INDEX -> (algo, layout).  One task = one algo on one layout, looping
#   its policy seeds and all FOVs inside.
#     tasks  0-6   sp      on the 7 layouts   (11 policy seeds each)
#     tasks  7-13  e3t                        ( 5 policy seeds each)
#     tasks 14-20  sp_eps                     ( 5 policy seeds each)
#
#   sbatch --array=0-20%14 run_filter_eval.sbatch          # everything
#   FOVS=30,90,360 EPISODES=5 PSEEDS=3 sbatch --array=0 ... # a fast probe
#
# Collisions are OFF unconditionally (filter/baseline.py::Kitchen), matching the
# ZSC eval's _nocoll arm, so these rows are comparable to eval/results/*_nocoll.
#
# fov120 is omitted ON PURPOSE: ZSC_BASELINES.md measured fov90 and fov120 as
# 98% bit-identical, so the axis has 5 levels, not 6.
# =============================================================================
set -u
cd "$HOME/steakhouse"
mkdir -p logs
source ~/.bashrc 2>/dev/null || true
conda activate steakhouse-ai
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/ 2>/dev/null || true

LAYOUTS=(steak_gc00 steak_gc03 steak_gc04 steak_gc06 steak_cram steak_cram2 steak_gs00)
ALGOS=(sp e3t sp_eps)

T=${SLURM_ARRAY_TASK_ID:-0}
ALGO=${ALGOS[$((T / 7))]}
LAYOUT=${LAYOUTS[$((T % 7))]}

# SP trained seeds 5-15; E3T and SP-eps trained seeds 1-5
if [ "$ALGO" = "sp" ]; then FIRST=5; NMAX=11; else FIRST=1; NMAX=5; fi
NP=${PSEEDS:-$NMAX}                       # how many policy seeds to use
EPISODES=${EPISODES:-10}
FOVS=${FOVS:-30,60,90,180,360}
K=${K:-3}; DEPTH=${DEPTH:-5}; ALPHA=${ALPHA:-0.5}

OUT=/scratch1/$USER/steakhouse_filter/${ALGO}_${LAYOUT}.jsonl
mkdir -p "$(dirname "$OUT")"
# rows files are APPEND-ONLY -- clear first or two runs silently mix
rm -f "$OUT"

echo "[task $T] algo=$ALGO layout=$LAYOUT pseeds=$FIRST..$((FIRST+NP-1)) fovs=$FOVS eps=$EPISODES k=$K d=$DEPTH a=$ALPHA"
for ((i=0; i<NP; i++)); do
  S=$((FIRST + i))
  CK=/scratch1/$USER/steakhouse_zsc/$ALGO/${LAYOUT}_seed${S}/sp_${LAYOUT}.pt
  [ -f "$CK" ] || { echo "  skip seed $S (no checkpoint)"; continue; }
  echo "=== $ALGO $LAYOUT policy-seed $S ==="
  python fov/robot/policy/new/filter/play_episode.py \
      --layout "$LAYOUT" --algo "$ALGO" --pseed "$S" \
      --fovs "$FOVS" --episodes "$EPISODES" \
      --k "$K" --depth "$DEPTH" --alpha "$ALPHA" \
      --device cpu --out "$OUT" || echo "  FAILED seed $S"
done
echo "[done] $OUT  rows=$(wc -l < "$OUT" 2>/dev/null || echo 0)"
