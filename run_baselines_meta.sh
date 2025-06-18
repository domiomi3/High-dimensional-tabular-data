#!/bin/bash 
#SBATCH --partition=mlhiwidlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --job-name=qsar_all
#SBATCH --time=1:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --export=ALL
#SBATCH --output=/work/dlclarge2/matusd-toy_example/LOGS//%x.%N.%A.%a.out
#SBATCH --error=/work/dlclarge2/matusd-toy_example/LOGS//%x.%N.%A.%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matus.dominika@gmail.com
#SBATCH --array=0-9

DATASET_ID=$1
N_FEATURES=$2
METRIC_OVERRIDE=$3

WORKING_DIR=/work/dlclarge2/matusd-toy_example 
CONDA_BASE=/home/matusd/.conda/
ENV_NAME=tabpfn
PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PYTHON_SCRIPT="$WORKING_DIR/train_sklearn.py"

source "$CONDA_BASE"/bin/activate "$ENV_NAME"

methods=(original random_fs variance_fs tree_fs kbest_fs pca_dr random_dr kpca_dr agglo_dr ica_dr)
METHOD=${methods[$SLURM_ARRAY_TASK_ID]}

echo "Running method $METHOD on dataset $DATASET"

cd $WORKING_DIR

if [ -n "$METRIC_OVERRIDE" ]; then
  "$PYTHON_BIN" "$PYTHON_SCRIPT" --method "$METHOD" --dataset "$DATASET_ID" --metric_override "$METRIC_OVERRIDE" 
else
  "$PYTHON_BIN" "$PYTHON_SCRIPT" --method "$METHOD" --dataset "$DATASET_ID" --n_features $N_FEATURES
fi