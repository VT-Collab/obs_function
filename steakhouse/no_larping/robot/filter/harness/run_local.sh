#!/bin/bash
# Run the grid across local cores, one process per (layout, fov) cell.
#
#   ./run_local.sh OUTDIR NPROC "method,method,..." "layouts" "fovs" "seeds"
#
# One process per cell needs no shared file, so there is no interleaved-write
# corruption and a single failed cell can be re-run on its own -- the same
# reason run_carc.sbatch is a 30-task array. Use this for a pilot; use CARC for
# a full grid (~1.5 h wall-clock there against most of a day here).
#
# If ssh to CARC refuses a non-PTY exec channel, or `discovery.usc.edu` will not
# resolve even with the VPN up, see the CARC note in RESULTS.md -- the internal
# nameserver has the host as 10.72.0.13 and
# `ssh -o HostKeyAlias=discovery.usc.edu 10.72.0.13` works when the name does not.
set -u

# KILL THE WHOLE TREE ON EXIT. Without this, stopping the wrapper leaves xargs
# and its workers running: they get reparented to init and keep burning CPU and
# memory for the rest of the session. Three abandoned grids' worth of those is
# what turned a 55s episode into a 20-minute one and looked, from the outside,
# exactly like the machine being slow.
cleanup() {
  trap - EXIT INT TERM
  pkill -P $$ 2>/dev/null
  kill -- -$$ 2>/dev/null
}
trap cleanup EXIT INT TERM

OUT=${1:?outdir}
NP=${2:-8}
METHODS=${3:-"handoff,qmdp"}
LAYOUTS=${4:-"back_bar banquet_pass butchery chefs_table divide pantry"}
FOVS=${5:-"30 60 90 180 360"}
SEEDS=${6:-"0-9"}
PY=${PY:-/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python}

mkdir -p "$OUT"
cd "$(dirname "$0")/../../.." || exit 1

for L in $LAYOUTS; do
  for F in $FOVS; do
    echo "$L $F"
  done
done | xargs -P "$NP" -n 2 bash -c '
  L=$0; F=$1
  out="'"$OUT"'/${L}_fov${F}.jsonl"
  [ -s "$out" ] && { echo "skip $L fov$F (exists)"; exit 0; }
  '"$PY"' -m robot.filter.harness.evaluate --layouts "$L" --fovs "$F" --seeds "'"$SEEDS"'" \
      --methods "'"$METHODS"'" --horizon 400 --out "$out" --quiet \
      && echo "done $L fov$F" || echo "FAIL $L fov$F"
'
echo "=== all cells finished -> $OUT ==="
