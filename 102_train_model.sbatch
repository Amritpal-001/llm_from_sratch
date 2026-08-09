#!/bin/bash
#SBATCH -J llmfs-train
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
./run.sh 102_train_model.py
