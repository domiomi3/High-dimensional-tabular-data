#!/usr/bin/env bash
# Create Slurm scripts for every combination of {models} x {datasets} with a methods set.
# Positionals (in this order):
#   1) MODELS   – space-separated list in quotes OR 'all'
#   2) DATASETS – space-separated list in quotes OR 'all'
#                 (tokens can be aliases: bioresponse|hiva|qsar OR numeric OpenML IDs OR csv:path)
#   3) METHODS  – space-separated list in quotes OR 'all'
#
# Any args after a literal `--` are forwarded verbatim to the Python generator,
# which will in turn forward unknown flags to train.py (e.g. --num_features 123).
#
# Examples:
#   ./generate_slurm_scripts.sh "tabpfnv2_tab catboost_tab" all "kbest_fs random_fs" -- --num_features 123 --seed 324
#   ./generate_slurm_scripts.sh all all all -g final -c -- --fs_ratio 0.6
#   ./generate_slurm_scripts.sh "tabpfnv2_tab" "hiva 363697" all -t -g debug -r 20241229_143000 -- --log_level DEBUG
#   ./generate_slurm_scripts.sh "tabpfnv2_tab" "csv:/path/to/data.csv" all -g csv_test -- --num_gpus 1 --num_cpus 4
#
set -euo pipefail

# ---------- defaults / config ----------
PYGEN="experiments/generate_slurm_script.py"

# dataset alias -> OpenML task id
declare -A DATASET_IDS=(
  [bioresponse]=363620
  [hiva]=363677
  [qsar]=363697
)

# canonical "all"
ALL_MODELS=("tabpfnv2_tab" "catboost_tab")
ALL_DATASETS=("bioresponse" "hiva" "qsar")

# flags for the generator (these are *recognized* by your Python script)
DRY_RUN=false
SAVE_TIME=false
EXP_GROUP="default"
RUN_ID=""
PARTITION=""
TIME_LIMIT=""
MAIL_USER=""
WORK_DIR=""
VENV_DIR=""
PYTHON_EXE=""

# ---------- parse options before the `--` separator ----------
POSITIONALS=()
FORWARD_ARGS=()
SEEN_DASHDASH=false

while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    SEEN_DASHDASH=true
    shift
    break
  fi
  case "$1" in
    -t|--dry_run) DRY_RUN=true; shift ;;
    -c|--save_time) SAVE_TIME=true; shift ;;
    -g|--group) EXP_GROUP="${2:-}"; shift 2 ;;
    -r|--run_id) RUN_ID="${2:-}"; shift 2 ;;
    -p|--partition) PARTITION="${2:-}"; shift 2 ;;
    --time) TIME_LIMIT="${2:-}"; shift 2 ;;
    --mail_user) MAIL_USER="${2:-}"; shift 2 ;;
    --working_dir) WORK_DIR="${2:-}"; shift 2 ;;
    --venv_dir) VENV_DIR="${2:-}"; shift 2 ;;
    --python_exe) PYTHON_EXE="${2:-}"; shift 2 ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  generate_all.sh "<models|all>" "<datasets|all>" "<methods|all>" [options] -- [forwarded args]

For datasets, you can specify:
  - 'all' for all predefined datasets
  - Space-separated OpenML task IDs (e.g., "363620 363677")
  - Space-separated aliases (e.g., "bioresponse hiva")
  - CSV paths prefixed with 'csv:' (e.g., "csv:/path/to/data.csv")
  - Mix of the above (e.g., "bioresponse csv:/data/custom.csv 363697")

Options (handled here, passed to the generator accordingly):
  -t, --dry_run             Use --dry_run in Python script (short wallclock)
  -c, --save_time           Use --save_time in Python script
  -g, --group NAME          Experiment group (default: default)
  -r, --run_id ID           Custom run ID (default: auto-generated timestamp)
  -p, --partition NAME      Slurm partition (overrides generator default)
      --time HH:MM:SS       Slurm time limit (overrides generator computed value)
      --mail_user EMAIL     Slurm mail user
      --working_dir PATH    Override working_dir for generator
      --venv_dir PATH       Override venv_dir for generator
      --python_exe PATH     Override python_exe for generator

Anything after `--` is forwarded verbatim to the Python generator, which
forwards unknown args to train.py (e.g. --num_features 123 --random_state 324).

The run_id is particularly useful for coordinating multiple related experiments:
  # All these will use the same run_id, grouping results together:
  ./generate_slurm_scripts.sh "tabpfn" all "original" -r experiment_v1 
  ./generate_slurm_scripts.sh "catboost" all "original" -r experiment_v1
USAGE
      exit 0 ;;
    -*)
      echo "❌ Unknown option: $1" >&2
      exit 1 ;;
    *)
      POSITIONALS+=("$1"); shift ;;
  esac
done

# collect everything after `--` for forwarding
if [[ "$SEEN_DASHDASH" == true ]]; then
  while [[ $# -gt 0 ]]; do FORWARD_ARGS+=("$1"); shift; done
fi

present=false
for tok in "${FORWARD_ARGS[@]:-}"; do
  if [[ "$tok" == "--num_gpus" || "$tok" == --num_gpus=* ]]; then present=true; break; fi
done
if [[ "$present" == false ]]; then
  FORWARD_ARGS+=("--num_gpus" "1")
fi

present=false
for tok in "${FORWARD_ARGS[@]:-}"; do
  if [[ "$tok" == "--num_cpus" || "$tok" == --num_cpus=* ]]; then present=true; break; fi
done
if [[ "$present" == false ]]; then
  FORWARD_ARGS+=("--num_cpus" "4")
fi

# ---------- check positionals ----------
MODELS_ARG="${POSITIONALS[0]:-all}"
DATASETS_ARG="${POSITIONALS[1]:-all}"
METHODS_ARG="${POSITIONALS[2]:-all}"

# expand MODELS
if [[ "$MODELS_ARG" == "all" ]]; then
  MODELS=("${ALL_MODELS[@]}")
else
  # split on spaces
  read -r -a MODELS <<<"$MODELS_ARG"
fi

# expand DATASETS
declare -a DATASETS_TOKENS
if [[ "$DATASETS_ARG" == "all" ]]; then
  DATASETS_TOKENS=("${ALL_DATASETS[@]}")
else
  read -r -a DATASETS_TOKENS <<<"$DATASETS_ARG"
fi

# resolve datasets to OpenML IDs or CSV paths
declare -a DATASET_ARGS=()
for tok in "${DATASETS_TOKENS[@]}"; do
  if [[ "$tok" =~ ^csv: ]]; then
    # Strip 'csv:' prefix and add as csv_path argument
    csv_file="${tok#csv:}"
    DATASET_ARGS+=("csv_path:$csv_file")
  elif [[ "$tok" =~ ^[0-9]+$ ]]; then
    DATASET_ARGS+=("openml_id:$tok")
  else
    if [[ -v "DATASET_IDS[$tok]" ]]; then
      DATASET_ARGS+=("openml_id:${DATASET_IDS[$tok]}")
    else
      echo "❌ Unknown dataset alias '$tok'. Allowed: ${!DATASET_IDS[*]} or numeric ID or 'csv:/path' or 'all'." >&2
      exit 1
    fi
  fi
done

# methods string is passed as-is (could be "all" or a quoted list)
METHODS_STR="$METHODS_ARG"

# sanity: at least one dataset arg if not "all"
if [[ "${#DATASET_ARGS[@]}" -eq 0 ]]; then
  echo "❌ No datasets resolved. Check your DATASETS argument." >&2
  exit 1
fi
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "❌ No models given." >&2
  exit 1
fi

# ---------- build common flags for the generator ----------
COMMON_FLAGS=( "--methods" "$METHODS_STR" "--exp_group" "$EXP_GROUP" )
[[ "$DRY_RUN" == true ]]   && COMMON_FLAGS+=( "--dry_run" )
[[ "$SAVE_TIME" == true ]] && COMMON_FLAGS+=( "--save_time" )
[[ -n "$RUN_ID" ]]         && COMMON_FLAGS+=( "--run_id" "$RUN_ID" )
[[ -n "$PARTITION" ]]      && COMMON_FLAGS+=( "--partition" "$PARTITION" )
[[ -n "$TIME_LIMIT" ]]     && COMMON_FLAGS+=( "--time" "$TIME_LIMIT" )
[[ -n "$MAIL_USER" ]]      && COMMON_FLAGS+=( "--mail_user" "$MAIL_USER" )
[[ -n "$WORK_DIR" ]]       && COMMON_FLAGS+=( "--working_dir" "$WORK_DIR" )
[[ -n "$VENV_DIR" ]]       && COMMON_FLAGS+=( "--venv_dir" "$VENV_DIR" )
[[ -n "$PYTHON_EXE" ]]     && COMMON_FLAGS+=( "--python_exe" "$PYTHON_EXE" )

# ---------- show run info ----------
if [[ -n "$RUN_ID" ]]; then
  echo "🏷️  Using custom run ID: $RUN_ID"
else
  echo "🏷️  Run ID will be auto-generated (timestamp)"
fi

# ---------- loop ----------
for data_arg in "${DATASET_ARGS[@]}"; do
  for mdl in "${MODELS[@]}"; do
    if [[ "$data_arg" =~ ^openml_id: ]]; then
      oid="${data_arg#openml_id:}"
      echo "→ Generating for model='$mdl' openml_id='$oid' methods='$METHODS_STR'"
      python "$PYGEN" \
        --openml_id "$oid" \
        --model "$mdl" \
        "${COMMON_FLAGS[@]}" \
        "${FORWARD_ARGS[@]}"
    else  # csv_path
      csv_path="${data_arg#csv_path:}"
      echo "→ Generating for model='$mdl' csv_path='$csv_path' methods='$METHODS_STR'"
      python "$PYGEN" \
        --csv_path "$csv_path" \
        --model "$mdl" \
        "${COMMON_FLAGS[@]}" \
        "${FORWARD_ARGS[@]}"
    fi
  done
done

echo "✓ All scripts generated."