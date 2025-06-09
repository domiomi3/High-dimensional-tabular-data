#!/bin/bash 
#SBATCH --partition=mlhiwidlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --job-name=baselines_all_qsar-tid-11
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --export=ALL
#SBATCH --output=/work/dlclarge2/matusd-toy_example/LOGS//%x.%N.%A.%a.out
#SBATCH --error=/work/dlclarge2/matusd-toy_example/LOGS//%x.%N.%A.%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matus.dominika@gmail.com
#SBATCH --array=0-6

WORKING_DIR=/work/dlclarge2/matusd-toy_example 
CONDA_BASE=/home/matusd/.conda/
ENV_NAME=tabpfn
PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PYTHON_SCRIPT="$WORKING_DIR/sklearn_baselines.py"

source "$CONDA_BASE"/bin/activate "$ENV_NAME"

methods=(original random_fs variance_fs tree_fs pca_dr random_dr agglo_dr)
METHOD=${methods[$SLURM_ARRAY_TASK_ID]}

echo "Running method: $METHOD"

cd $WORKING_DIR
"$PYTHON_BIN" "$PYTHON_SCRIPT" --method "$METHOD" 