#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Cluster defaults
# ---------------------------------------------------------------------------
DEF_WORK_DIR  = "/work/dlclarge2/matusd-toy_example"
DEF_PARTITION = "mlhiwidlc_gpu-rtx2080"
DEF_TIME      = "2:00:00"
DEF_CONDA     = "$HOME/.conda"

# ---------------------------------------------------------------------------
# utils.load_dataset helper
# ---------------------------------------------------------------------------
UTILS_DIR = "/work/dlclarge2/matusd-toy_example/src"
sys.path.append(UTILS_DIR)
try:
    from utils.openml_data import load_dataset                                   # noqa: E402
except Exception as e:
    sys.exit(f"❌ Could not import load_dataset from {UTILS_DIR}: {e}")

# ---------------------------------------------------------------------------
ALLOWED_METHODS = (
    "original random_fs variance_fs tree_fs kbest_fs "
    "pca_dr random_dr kpca_dr agglo_dr"
).split()

# ---------------------------------------------------------------------------
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --export=ALL
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
{mail_line}{array_line}

OPENML_ID={openml_id}
MODEL={model}

WORKING_DIR={working_dir}
CONDA_BASE={conda_base}
ENV_NAME=tabpfn
PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PYTHON_SCRIPT="$WORKING_DIR/src/train.py"

source "$CONDA_BASE"/bin/activate "$ENV_NAME"

methods=({methods_array})
METHOD=${{methods[$SLURM_ARRAY_TASK_ID]}}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)         
LOG_PREFIX="{script_base}_${{METHOD}}_${{TIMESTAMP}}"
LOG_ID="${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"

exec >  "{log_dir}/${{LOG_PREFIX}}.${{LOG_ID}}.out" \
     2> "{log_dir}/${{LOG_PREFIX}}.${{LOG_ID}}.err"

echo "Running method ${{METHOD}} with ${{MODEL}} on OpenML task ${{OPENML_ID}}"

cd "$WORKING_DIR"

RESULTS_DIR="{results_root}/${{METHOD}}"
mkdir -p "$RESULTS_DIR"

"$PYTHON_BIN" "$PYTHON_SCRIPT" \\
    --method "$METHOD" \\
    --openml_id "$OPENML_ID"{dry_run_flag}{check_time_flag} \\
    --model "$MODEL" \\
    --results_dir "$RESULTS_DIR"
"""

# ---------------------------------------------------------------------------
def parse_methods(arg: str) -> List[str]:
    if arg.lower() in {"all", "*"}:
        return ALLOWED_METHODS.copy()
    lst = [m.strip() for m in arg.split(" ") if m.strip()]
    bad = [m for m in lst if m not in ALLOWED_METHODS]
    if bad:
        raise ValueError(f"Unknown method(s): {', '.join(bad)}")
    return lst


def clean(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def make_abbr(methods: List[str]) -> str:
    if set(methods) == set(ALLOWED_METHODS):
        return "all"
    parts = []
    if "original" in methods:
        parts.append("o")
    fs = sorted(m for m in methods if m.endswith("_fs"))
    if fs:
        parts.append("fs_" + "".join(m[0] for m in fs))
    dr = sorted(m for m in methods if m.endswith("_dr"))
    if dr:
        parts.append("dr_" + "".join(m[0] for m in dr))
    return "_".join(parts)


def safe_load_dataset(oid: int) -> str:
    obj = load_dataset(oid)
    return obj[1].name if isinstance(obj, tuple) else obj.name


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a single Slurm array script for TabPFN/OpenML runs.")
    ap.add_argument("--openml_id", type=int, required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--exp_group", required=True)
    ap.add_argument("--model", default="tabpfnv2_org")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--check_time", action="store_true")
    # cluster knobs
    ap.add_argument("--working_dir", default=DEF_WORK_DIR)
    ap.add_argument("--conda_base",  default=DEF_CONDA)
    ap.add_argument("--partition",   default=DEF_PARTITION)
    ap.add_argument("--time",        default=DEF_TIME)
    ap.add_argument("--mail_user")
    args = ap.parse_args()

    is_tabpfn = args.model.lower().startswith("tabpfn")

    if args.test:                                # dry-run jobs
        time_str = "0:25:00"  if is_tabpfn else "1:15:00"
    else:                                         # full runs
        time_str = "2:00:00"  if is_tabpfn else "6:00:00"
        
    methods = parse_methods(args.methods)

    try:
        dataset_raw = safe_load_dataset(args.openml_id)
        dataset_name = clean(dataset_raw)
    except Exception as e:
        sys.exit(f"❌ Could not load dataset: {e}")

    # base directories -------------------------------------------------------
    base = Path(args.working_dir, "experiments")
    logs_dir     = base / "slurm_logs"    / args.exp_group / dataset_name / args.model
    scripts_dir  = base / "slurm_scripts" / args.exp_group / dataset_name / args.model
    results_root = base / "results"       / args.exp_group / dataset_name / args.model

    for d in (logs_dir, scripts_dir, results_root):
        d.mkdir(parents=True, exist_ok=True)

    abbr  = make_abbr(methods)
    extra = "_test" if args.test else ""
    script_base = f"{dataset_name}_{args.model}{extra}"
    script_path = scripts_dir / f"{script_base}_{abbr}.sh"

    # flags & header helpers --------------------------------------------------
    array_line     = "#SBATCH --array=0" if len(methods) == 1 \
                     else f"#SBATCH --array=0-{len(methods)-1}"
    dry_run_flag    = " \\\n    --dry_run"    if args.test        else ""
    check_time_flag = " \\\n    --check_time" if args.check_time  else ""
    mail_line = (f"#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={args.mail_user}"
                 if args.mail_user else "")

    slurm_text = SLURM_TEMPLATE.format(
        partition       = args.partition,
        time            = time_str,
        job_name        = f"{args.model}_{dataset_name}",
        log_dir         = logs_dir,
        script_base     = script_base,
        mail_line       = mail_line,
        array_line      = array_line,
        working_dir     = args.working_dir,
        conda_base      = args.conda_base,
        methods_array   = " ".join(methods),
        results_root    = results_root,
        model           = args.model,
        dry_run_flag    = dry_run_flag,
        check_time_flag = check_time_flag,
        openml_id       = args.openml_id,
    )

    with script_path.open("w") as f:
        f.write(dedent(slurm_text))
    os.chmod(script_path, 0o755)

    print(f"✓ Slurm script written to: {script_path}")
    print("  Submit with:  sbatch", script_path)


if __name__ == "__main__":
    main()
