#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --job-name=average_rmse
#SBATCH --output=LOGS/average_rmse.%N.%j.out
#SBATCH --error=LOGS/average_rmse.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=matus.dominika@gmail.com
#SBATCH --time=0:05:00
#SBATCH --mem=2gb

echo "Running RMSE averaging..."

source /pfs/data6/home/fr/fr_fr/fr_dm339/miniconda3/bin/activate tabpfn

python utils.py
