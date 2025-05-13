#!/bin/bash 
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --job-name=baselines
#SBATCH --time=10:00
#SBATCH --mem=15gb
#SBATCH --output=LOGS//%x.%N.%A.%a.out
#SBATCH --error=LOGS//%x.%N.%A.%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=matus.dominika@gmail.com
#SBATCH --array=0-2

IFS=' ' read -r -a SEEDS <<< "$SEEDS_ARRAY"
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Running with seed: $SEED"

source /pfs/data6/home/fr/fr_fr/fr_dm339/miniconda3/bin/activate tabpfn

mkdir -p results

python sklearn_baselines.py \
    --dataset dataset \
    --method original random_fs variance_fs tree_fs pca_dr random_dr agglo_dr \
    --random_state $SEED \
    > results/results_seed_${SEED}.txt
