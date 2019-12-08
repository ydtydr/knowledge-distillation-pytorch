#!/bin/bash
#SBATCH -J kd
#SBATCH -o kd.o%j
#SBATCH -e kd.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=20000
#SBATCH -t 24:00:00
#SBATCH --partition=desa  --gres=gpu:1

cd /home/ty367/knowledge-distillation-pytorch/
# python train.py --model_dir experiments/base_resnet18_vanila
python train.py --model_dir experiments/resnet18_distill/resnext_teacher