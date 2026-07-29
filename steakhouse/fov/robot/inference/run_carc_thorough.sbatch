#!/bin/bash -l
#SBATCH --job-name=fov_thorough
#SBATCH --account=biyik_1173
#SBATCH --partition=main
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -e
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate steakhouse-ai
cd ~/steakhouse
echo "########## 1. CHEAT AUDIT across layouts, 6 FOVs, 10 seeds ##########"
for L in steak_side_2 steak_island steak_api steak_test steak_none_3 steak_parrallel steak_side_3; do
  echo "--- $L ---"
  python -m fov.human.test_no_cheating $L 10 2>/dev/null | grep -E "^ *[0-9]+ |verdict|PASS|FAIL|CHEAT"
done
echo "########## 2. INFERENCE, recommended triples, 15 seeds ##########"
python -m fov.robot.inference.evaluate_inference 32 15
echo "########## 3. INFERENCE, HARDER 6-way hypothesis set, 15 seeds ##########"
python -m fov.robot.inference.evaluate_inference 32 15 full
