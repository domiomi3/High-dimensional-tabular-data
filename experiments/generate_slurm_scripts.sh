#!/usr/bin/env bash
# --------------------------------------------------------------------------
# generate_slurm_scripts.sh
#
# Usage examples
#   ./generate_slurm_scripts.sh --group final         # ALL datasets/models/methods for "final" experiment
#   ./generate_slurm_scripts.sh hiva --group final    # HIVA, all models/methods for "final" experiment
#   ./generate_slurm_scripts.sh hiva tabpfnv2_org     # one dataset, one model for "default" experiment
#   ./generate_slurm_scripts.sh hiva tabpfnv2_org "kbest_fs random_fs" # one dataset, one model, 
#   ./generate_slurm_scripts.sh --test --check-time --group debug  # dry-run variants for every job with time check for "debug" experiment
#
# Positional args
#   1) DATASET   – bioresponse | hiva | qsar | all
#   2) MODEL(S)  – space-separated list OR all
#   3) METHOD(S) – space-separated list OR all
# Optional flags
#   -t | --test        – pass --test to generate_slurm_script.py (dry-run)
#   -g | --group       – choose experiment-group directory (default)
#   -c | --check-time  – add --check_time to generate_slurm_script.py
# --------------------------------------------------------------------------
set -euo pipefail

# ── 0. flag parsing --------------------------------------------------------
TEST_MODE=false
CHECK_TIME=false
EXP_GROUP=default
POSITIONALS=()                     # will hold dataset / model / methods

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test)
            TEST_MODE=true
            shift ;;
        -g|--group)
            [[ ${2:-} ]] || { echo "❌ --group needs a value" >&2; exit 1; }
            EXP_GROUP=$2
            shift 2 ;;
        -c|--check-time)
            CHECK_TIME=true
            shift ;;
        --)                         # explicit end-of-options
            shift; break ;;
        -*)
            echo "❌ Unknown option: $1" >&2
            exit 1 ;;
        *)                          # positional: push to array and continue
            POSITIONALS+=("$1")
            shift ;;
    esac
done

# restore the (up to) three positionals so the rest of the script is unchanged
set -- "${POSITIONALS[@]}"

# ── 1. lookup table: dataset → OpenML task id --------------------------------
declare -A DATASET_IDS=(
    [bioresponse]=363620
    [hiva]=363677
    [qsar]=363697
)

# ── 2. full lists ------------------------------------------------------------
ALL_DATASETS=(bioresponse hiva qsar)
ALL_MODELS=(catboost_tab tabpfnv2_org tabpfnv2_tab)
ALL_METHODS="all"          # literal string passed through

# ── 3. positional arguments --------------------------------------------------
DATASET_ARG=${1:-all}
MODELS_ARG=${2:-all}
METHODS_ARG=${3:-all}

# allow a legacy 4th positional "test"
[[ ${4:-} == test ]] && TEST_MODE=true

# datasets list
if [[ $DATASET_ARG == "all" ]]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=($DATASET_ARG)
fi

# models list
if [[ $MODELS_ARG == "all" ]]; then
    MODELS=("${ALL_MODELS[@]}")
else
    MODELS=($MODELS_ARG)
fi

METHODS=$METHODS_ARG   # keep single string (could be "all" or list)

# ── 4. sanity checks ---------------------------------------------------------
for ds in "${DATASETS[@]}"; do
    [[ -v "DATASET_IDS[$ds]" ]] || {
        echo "❌ Unknown dataset '$ds'. Allowed: ${!DATASET_IDS[*]} or 'all'." >&2
        exit 1
    }
done

for mdl in "${MODELS[@]}"; do
    case $mdl in
        catboost_tab|tabpfnv2_org|tabpfnv2_tab) : ;;
        *) echo "❌ Unknown model '$mdl'. Allowed: ${ALL_MODELS[*]} or 'all'." >&2; exit 1 ;;
    esac
done

# ── 5. loop & launch ---------------------------------------------------------
PY=experiments/generate_slurm_script.py
TEST_FLAG=$([[ $TEST_MODE == true ]] && echo "--test" || echo "")
CHECK_TIME_FLAG=$([[ $CHECK_TIME == true ]] && echo "--check_time"   || echo "")

for ds in "${DATASETS[@]}"; do
    OPENML_ID=${DATASET_IDS[$ds]}
    for mdl in "${MODELS[@]}"; do
        python "$PY" \
            --openml_id "$OPENML_ID" \
            --methods "$METHODS" \
            --exp_group "$EXP_GROUP" \
            --model "$mdl" \
            $TEST_FLAG \
            $CHECK_TIME_FLAG
    done
done

echo "✓ All generate_slurm_script.py commands issued."
