#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 #Using one GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
# Wall time
#SBATCH --time=72:00:00
#SBATCH --job-name=embrace_img
#SBATCH --output=/cluster/home/jessesun/emb_ic/200206_ConvLSTM.txt
# Emails me when job starts, ends or fails
#SBATCH --mail-user=j294sun@uwaterloo.ca.com
#SBATCH --mail-type=ALL

# activate the virtual environment
source /cluster/home/jessesun/anaconda3/bin/activate

# run a training session
srun python train.py --batch_size 10 --epoch 60 --lr 0.01 --optimizer sgd
