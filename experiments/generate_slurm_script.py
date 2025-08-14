#!/usr/bin/env python3
"""
Generate a Slurm array script for TabPFN/OpenML experiments (micromamba only).

- Uses micromamba exclusively (no conda).
- Activates env via `micromamba run` (no shell hook needed).
- Keeps your abbreviations, method parsing, and directory layout.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import List

# ---------------------------------------------------------------------------
# Cluster defaults
# ---------------------------------------------------------------------------
DEF_WORK_DIR   = "/work/dlclarge2/matusd-toy_example"
DEF_PARTITION  = "mlhiwidlc_gpu-rtx2080"
DEF_TIME       = "2:00:00"
DEF_MAMBA_ROOT = "$HOME/micromamba"
DEF_MAMBA_EXE  = "$HOME/bin/micromamba"

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
    "pca_dr random_dr kpca_dr agglo_dr kbest+pca"
).split()

# Abbreviation map for single-method runs or misc groups
_ABBR_MAP = {
    "original":     "orig",
    "kbest+pca":    "kbest+pca",
    "random_fs":    "rands",
    "variance_fs":  "var",
    "tree_fs":      "tree",
    "kbest_fs":     "kbest",
    "pca_dr":       "pca",
    "random_dr":    "randdr",
    "agglo_dr":     "agglo",
    "kpca_dr":      "kpca",
}

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
MAMBA_ROOT_PREFIX={mamba_root}
MAMBA_EXE={mamba_exe}
ENV_NAME=high_tab
PYTHON_SCRIPT="$WORKING_DIR/src/train.py"

methods=({methods_array})
METHOD=${{methods[$SLURM_ARRAY_TASK_ID]}}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PREFIX="{script_base}_${{METHOD}}_${{TIMESTAMP}}"
LOG_ID="${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"

# Ensure log/result dirs exist and redirect logs
exec >  "{log_dir}/${{LOG_PREFIX}}.${{LOG_ID}}.out" \
     2> "{log_dir}/${{LOG_PREFIX}}.${{LOG_ID}}.err"

echo "Running method ${{METHOD}} with ${{MODEL}} on OpenML task ${{OPENML_ID}}"

cd "$WORKING_DIR"

RESULTS_DIR="{results_root}/${{METHOD}}"
mkdir -p "$RESULTS_DIR"

# Sanity-check micromamba binary
if [[ ! -x "$MAMBA_EXE" ]]; then
  echo "❌ micromamba not found or not executable at $MAMBA_EXE" >&2
  exit 1
fi

# Run without activating a login shell; use the env path directly
"$MAMBA_EXE" run -p "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME" python "$PYTHON_SCRIPT" \\
    --method "$METHOD" \\
    --openml_id "$OPENML_ID"{dry_run_flag}{check_time_flag} \\
    --model "$MODEL" \\
    --results_dir "$RESULTS_DIR"
"""

# ---------------------------------------------------------------------------
def parse_methods(arg: str) -> List[str]:
    """Return validated list of preprocessing methods."""
    if arg.lower() in {"all", "*"}:
        return ALLOWED_METHODS.copy()
    lst = [m.strip() for m in arg.split(" ") if m.strip()]
    bad = [m for m in lst if m not in ALLOWED_METHODS]
    if bad:
        raise ValueError(f"Unknown method(s): {', '.join(bad)}")
    return lst


def clean(s: str) -> str:
    """File-system–friendly slug."""
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def make_abbr(methods: List[str]) -> str:
    """Short slug that encodes which preprocessing methods are used."""
    # 1) full sweep
    if set(methods) == set(ALLOWED_METHODS):
        return "all"

    # 2) single-method run
    if len(methods) == 1:
        m = methods[0]
        return _ABBR_MAP.get(m, re.sub(r"[^a-z]", "", m)[:3])

    # 3) mixed run – bucket logic
    parts = []
    if "original" in methods:
        parts.append("o")

    fs = sorted(m for m in methods if m.endswith("_fs"))
    if fs:
        parts.append("fs_" + "".join(m[0] for m in fs))

    dr = sorted(m for m in methods if m.endswith("_dr"))
    if dr:
        parts.append("dr_" + "".join(m[0] for m in dr))

    misc = sorted(m for m in methods if m not in fs + dr + ["original"])
    if misc:
        parts.append("x_" + "".join(_ABBR_MAP.get(m, m)[0] for m in misc))

    return "_".join(parts)


def safe_load_dataset(oid: int) -> str:
    obj = load_dataset(oid)
    return obj[1].name if isinstance(obj, tuple) else obj.name


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a single Slurm array script for TabPFN/OpenML runs (micromamba only).")
    ap.add_argument("--openml_id", type=int, required=True)
    ap.add_argument("--methods",   required=True)
    ap.add_argument("--exp_group", required=True)
    ap.add_argument("--model",     required=True)
    ap.add_argument("--test",      action="store_true")
    ap.add_argument("--check_time", action="store_true")
    # cluster knobs
    ap.add_argument("--working_dir", default=DEF_WORK_DIR)
    ap.add_argument("--partition",   default=DEF_PARTITION)
    ap.add_argument("--time",        default=DEF_TIME)
    ap.add_argument("--mail_user")
    # micromamba knobs
    ap.add_argument("--mamba_root", default=DEF_MAMBA_ROOT,
                    help="Micromamba root prefix (contains envs/)")
    ap.add_argument("--mamba_exe",  default=DEF_MAMBA_EXE,
                    help="Path to micromamba executable")
    args = ap.parse_args()

    is_tabpfn = args.model.lower().startswith("tabpfn")

    # wall-time presets
    time_str = ("0:25:00" if is_tabpfn else "1:15:00") if args.test \
               else ("2:00:00" if is_tabpfn else "10:00:00")

    methods = parse_methods(args.methods)

    try:
        dataset_raw  = safe_load_dataset(args.openml_id)
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

    abbr = make_abbr(methods)
    extra = "_test" if args.test else ""
    script_base  = f"{dataset_name}_{args.model}{extra}"
    script_path  = scripts_dir / f"{script_base}{'_' + abbr if abbr else ''}.sh"

    # flags & header helpers --------------------------------------------------
    array_line = "#SBATCH --array=0" if len(methods) == 1 \
                 else f"#SBATCH --array=0-{len(methods)-1}"
    dry_run_flag    = " \\\n    --dry_run"    if args.test       else ""
    check_time_flag = " \\\n    --check_time" if args.check_time else ""
    mail_line = (f"#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={args.mail_user}\n"
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
        methods_array   = " ".join(methods),
        results_root    = results_root,
        model           = args.model,
        dry_run_flag    = dry_run_flag,
        check_time_flag = check_time_flag,
        openml_id       = args.openml_id,
        mamba_root      = args.mamba_root,
        mamba_exe       = args.mamba_exe,
    )

    with script_path.open("w") as f:
        f.write(dedent(slurm_text))
    os.chmod(script_path, 0o755)

    print(f"✓ Slurm script written to: {script_path}")
    print("  Submit with:  sbatch", script_path)


if __name__ == "__main__":
    main()
