#!/bin/bash

SEEDS=(1 44 75 100 241 529 996 1880 5456 98732)
SEED_STRING="${SEEDS[@]}"

echo $SEED_STRING > used_seeds.txt

NUM_SEEDS=${#SEEDS[@]}
ARRAY_END=$((NUM_SEEDS - 1))

export SEEDS_ARRAY="${SEED_STRING}"

BASELINE_JOB_ID=$(sbatch --parsable --array=0-${ARRAY_END} run_baselines.sh)
echo "Sklearn baseline SLURM job ID: $BASELINE_JOB_ID"

echo "RMSE averaging"
sbatch --dependency=afterok:${BASELINE_JOB_ID} run_average.sh
