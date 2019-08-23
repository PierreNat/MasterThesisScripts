#!/bin/bash
#SBATCH --mail-type=none
#SBATCH --job-name="180_6para"
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --constraint=rtx2080
# Your code below this line

python3 CNN_resnet50TrainValRenderV2.py
