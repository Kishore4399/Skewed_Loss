#!/bin/bash

#SBATCH --job-name=Skewed
#SBATCH --partition=defq
#SBATCH --time=999:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=100G
#SBATCH --output=./log/output/Skewed.log
#SBATCH --error=./log/error/Skewed.log

# python main.py --mode skewed \
#                 --epochs 90 \
#                 --batch-size 256 \
#                 --lr 1e-3 \
#                 --lambda-stage2-start 30 \
#                 --lambda-stage3-start 60 \
#                 --lambda-interval 3 \
#                 --save-start 20 \
#                 --lr-step 30 \
#                 --amp \
#                 --seed 4 \
#                 --dataset imdb_wiki \
#                 --data-folder /isilon/datalake/dma_research/scratch/damalab/Tools/imbalanced-regression/imdb-wiki-dir/data \
#                 --csv /isilon/datalake/dma_research/scratch/damalab/Tools/Rank-N-Contrast/data/imdb_wiki.csv \
#                 --label-name age \
#                 --model 'resnet18'


# Age DB
# python main.py --mode skewed \
#                 --epochs 300 \
#                 --lr 1e-3 \
#                 --lambda-stage2-start 100 \
#                 --lambda-stage3-start 200 \
#                 --lambda-interval 5 \
#                 --save-start 80 \
#                 --lr-step 50 \
#                 --seed 4 \
#                 --dataset agedb \
#                 --data-folder /isilon/datalake/dma_research/scratch/damalab/Data/Shihua/82bd6-main \
#                 --csv /isilon/datalake/dma_research/scratch/damalab/Tools/Rank-N-Contrast/data/agedb.csv \
#                 --label-name age \
#                 --model 'resnet18'

# SkyFinder
python main.py --mode skewed \
                --epochs 300 \
                --lr 1e-3 \
                --lambda-stage2-start 100 \
                --lambda-stage3-start 200 \
                --lambda-interval 5 \
                --save-start 80 \
                --lr-step 50 \
                --seed 0 \
                --dataset SkyFinder \
                --data-folder /isilon/datalake/dma_research/scratch/damalab/Data/Shihua/SkyFinder \
                --csv /isilon/datalake/dma_research/scratch/damalab/Data/Shihua/SkyFinder/skyfinder_random_split.csv \
                --label-name label \
                --model 'resnet18'