#!/bin/bash
# Brain age from 3D T1 MRI with the skewed loss, 10-fold cross-validation (UnbiasedBrainAge folds).
#
#   sbatch job_sub_brainage.sh                # all 10 folds, skewed and normal, seed 0
#   sbatch job_sub_brainage.sh 3              # only validation fold 3 (test fold (3 + 9) % 10 = 2)
#
# Then compare normal / two-stage / skewed over the folds:
#   python scripts/summarize.py --runs ./runs --dataset brainage

#SBATCH --job-name=Skewed_BrainAge
#SBATCH --partition=defq
#SBATCH --time=999:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=100G
#SBATCH --output=./log/output/%x_%j.log
#SBATCH --error=./log/error/%x_%j.log

# Fold CSV from UnbiasedBrainAge: columns image_path, age, fold
CSV='/path/to/UnbiasedBrainAge/dataset/brainage/UKB&ADNI_fold.csv'
# Relative image_path values are resolved against this folder (absolute paths are used as they are)
DATA_FOLDER='/path/to/UnbiasedBrainAge'
# Share the preprocessed-volume cache with UnbiasedBrainAge (same format), so volumes are not re-processed
export BRAINAGE_CACHE_DIR=/path/with/space/brainage_cache_96

SEED=0
FOLDS=${1:-"0 1 2 3 4 5 6 7 8 9"}

for v in $FOLDS; do
  t=$(((v + 9) % 10))
  for mode in skewed normal; do
    python main.py --modality mri --mode $mode \
                   --model resnet18_3d \
                   --dataset brainage \
                   --csv "$CSV" \
                   --data-folder "$DATA_FOLDER" \
                   --label-name age \
                   --val-fold $v --test-fold $t \
                   --image-size 96 96 96 \
                   --batch-size 64 \
                   --eval-batch-size 64 \
                   --workers 8 \
                   --epochs 200 \
                   --lr 1e-3 \
                   --lr-step 50 \
                   --lambda-stage2-start 60 \
                   --lambda-stage3-start 120 \
                   --lambda-interval 5 \
                   --save-start 50 \
                   --amp \
                   --seed $SEED
  done
done
