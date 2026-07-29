#!/bin/bash -l
# MISHA NEW CHANGE - re-measure everything against the corrected human, which now
# perceives its teammate through the SAME FOV+decay machinery as any station and
# therefore skips work it can SEE the robot already doing. That changes the
# human's policy, so inference accuracy and layout validation both had to be
# re-run rather than carried forward.
#SBATCH --job-name=fov_rerun_all
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
echo "######## 1. NO-CHEATING AUDIT ########"
python -m fov.human.test_no_cheating steak_side_2 10 2>/dev/null | grep -E "^ *[0-9]+ |PASS|FAIL|fov "
echo "######## 2. LAYOUT VALIDATION (both conditions) ########"
python -m fov.human.batch_validate 32 10
echo "######## 3. INFERENCE, triples, 15 seeds ########"
python -m fov.robot.inference.evaluate_inference 32 15
echo "######## 4. INFERENCE, 6-way, 15 seeds ########"
python -m fov.robot.inference.evaluate_inference 32 15 full
