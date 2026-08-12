#!/bin/bash
# One process per (layout, fov) cell, like run_local.sh. Kills its own tree on exit.
set -u
cleanup() { trap - EXIT INT TERM; pkill -P $$ 2>/dev/null; kill -- -$$ 2>/dev/null; }
trap cleanup EXIT INT TERM

OUT=${1:?outdir}
NP=${2:-8}
METHODS=${3:?methods}
SEEDS=${4:-"0-9"}
LAYOUTS=${5:-"back_bar banquet_pass butchery chefs_table divide pantry"}
FOVS=${6:-"30 60 90 180 360"}
PY=/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python

mkdir -p "$OUT"
cd "$(dirname "$0")/../../.." || exit 1

for L in $LAYOUTS; do for F in $FOVS; do echo "$L $F"; done; done \
| xargs -P "$NP" -n 2 bash -c '
  L=$0; F=$1
  out="'"$OUT"'/${L}_fov${F}.jsonl"
  [ -s "$out" ] && { echo "skip $L fov$F"; exit 0; }
  '"$PY"' -m robot.filter.dev.subtask_dev --layouts "$L" --fovs "$F" \
      --seeds "'"$SEEDS"'" --methods "'"$METHODS"'" --horizon 400 \
      --out "$out" --quiet >/dev/null 2>&1 \
      && echo "done $L fov$F" || echo "FAIL $L fov$F"
'
echo "=== finished -> $OUT ==="
