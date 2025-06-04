#!/bin/bash 
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --job-name=baselines_all_qsar-tid-11
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --export=ALL
#SBATCH --output=/pfs/work9/workspace/scratch/fr_dm339-toy_example/LOGS//%x.%N.%A.%a.out
#SBATCH --error=/pfs/work9/workspace/scratch/fr_dm339-toy_example/LOGS//%x.%N.%A.%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matus.dominika@gmail.com
#SBATCH --array=0-6

set -e
set -u
set -o pipefail
set -x

CONDA_BASE=/pfs/data6/home/fr/fr_fr/fr_dm339/miniconda3
ENV_NAME=tabpfn
PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"

source "$CONDA_BASE"/bin/activate "$ENV_NAME"

methods=(original random_fs variance_fs tree_fs pca_dr random_dr agglo_dr)
METHOD=${methods[$SLURM_ARRAY_TASK_ID]}

echo "Running method: $METHOD"

"$PYTHON_BIN" sklearn_baselines.py --method "$METHOD"