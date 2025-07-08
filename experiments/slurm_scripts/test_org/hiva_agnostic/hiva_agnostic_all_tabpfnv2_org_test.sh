#!/bin/bash
#SBATCH --partition=mlhiwidlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_org_hiva_agnostic
#SBATCH --time=1:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --export=ALL
#SBATCH --output=/work/dlclarge2/matusd-toy_example/experiments/slurm_logs/test_org/hiva_agnostic/hiva_agnostic_all_tabpfnv2_org_test.%N.%A.%a.out
#SBATCH --error=/work/dlclarge2/matusd-toy_example/experiments/slurm_logs/test_org/hiva_agnostic/hiva_agnostic_all_tabpfnv2_org_test.%N.%A.%a.err
#SBATCH --array=0-9

OPENML_ID=363677
MODEL=tabpfnv2_org

WORKING_DIR=/work/dlclarge2/matusd-toy_example
CONDA_BASE=$HOME/.conda
ENV_NAME=tabpfn
PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PYTHON_SCRIPT="$WORKING_DIR/src/train_sklearn.py"

source "$CONDA_BASE"/bin/activate "$ENV_NAME"

methods=(original random_fs variance_fs tree_fs kbest_fs pca_dr random_dr kpca_dr agglo_dr ica_dr)
METHOD=${methods[$SLURM_ARRAY_TASK_ID]}

echo "Running method ${METHOD} with ${MODEL} on OpenML task ${OPENML_ID}"

cd "$WORKING_DIR"

RESULTS_DIR="/work/dlclarge2/matusd-toy_example/experiments/results/test_org/hiva_agnostic/${METHOD}"
mkdir -p "$RESULTS_DIR"

"$PYTHON_BIN" "$PYTHON_SCRIPT" \
    --method "$METHOD" \
    --openml_id "$OPENML_ID" \
    --dry_run \
    --check_time \
    --model "$MODEL" \
    --results_dir "$RESULTS_DIR"
