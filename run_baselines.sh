#!/bin/bash 
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --job-name=baselines_all
#SBATCH --time=5:00:00
#SBATCH --mem=15gb
#SBATCH --output=LOGS//%x.%N.%A.%a.out
#SBATCH --error=LOGS//%x.%N.%A.%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=matus.dominika@gmail.com

source /pfs/data6/home/fr/fr_fr/fr_dm339/miniconda3/bin/activate tabpfn

python sklearn_baselines.py \
    --method original random_fs variance_fs tree_fs pca_dr random_dr agglo_dr 