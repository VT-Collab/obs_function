#!/bin/bash -l
# MISHA NEW CHANGE - full re-run against the NO-CHEATING agent.
#
# The agent changed materially: station locations must now be discovered by
# looking, tiles are not known walkable until seen, and all pathing is the
# agent's own BFS over seen floor (previously mlp.mp.get_plan searched the FULL
# map, so it routed optimally through rooms it had never entered). Every earlier
# layout-validation and inference number was measured with those advantages, so
# none of them describe this agent and all are being re-measured.
#SBATCH --job-name=fov_full_rerun
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse

echo "############ 1. NO-CHEATING AUDIT, 7 layouts x 6 FOVs x 10 seeds ############"
for L in steak_side_2 steak_island steak_api steak_test steak_none_3 steak_parrallel steak_side_3; do
  echo "===== $L ====="
  python -m fov.human.test_no_cheating $L 10 2>/dev/null | grep -E "^ *[0-9]+ |^[0-9]\.|PASS|FAIL|CHEAT AUDIT|fov "
done

echo "############ 2. LAYOUT RE-VALIDATION (both FOV conditions) ############"
python -m fov.human.batch_validate 32 10

echo "############ 3. INFERENCE, recommended triples, 15 seeds ############"
python -m fov.robot.inference.evaluate_inference 32 15

echo "############ 4. INFERENCE, HARDER 6-way hypothesis set, 15 seeds ############"
python -m fov.robot.inference.evaluate_inference 32 15 full
