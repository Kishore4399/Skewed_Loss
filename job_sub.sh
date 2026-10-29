#!/bin/bash

#SBATCH --job-name=Skewed
#SBATCH --partition=defq
#SBATCH --time=999:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=100G
#SBATCH --output=./log/output/Skewed.log
#SBATCH --error=./log/error/Skewed.log

python main.py --mode skewed \
                --epochs 300 \
                --lr 1e-3 \
                --lambda-stage2-start 100 \
                --lambda-stage3-start 200 \
                --lambda-interval 5 \
                --save-start 80 \
                --lr-step 50 \
                --seed 0 \
                --dataset agedb \
                --data-folder /isilon/datalake/dma_research/scratch/damalab/Tools/imbalanced-regression/imdb-wiki-dir/data \
                --csv /isilon/datalake/dma_research/scratch/damalab/Tools/Rank-N-Contrast/data/imdb_wiki.csv \
                --label-name age \
                --model 'resnet18'


# python postadpt.py -c configs/tta/PostTrainAdpt.yaml model.pretrained='results/resnet18_RnC_UKB&ADNI_Validation:0_Test:9/checkpoints/last.ckpt'