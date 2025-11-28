#!/usr/bin/env python3
import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import List

# -----------------------------------------------------------------------------
# Defaults / Paths
# -----------------------------------------------------------------------------
WORK_DIR = "/work/dlclarge2/matusd-toy_example"
VENV_DIR = f"{WORK_DIR}/.venv"
PYTHON_EXE = f"{VENV_DIR}/bin/python"
DEF_PARTITION = "mldlc2_gpu-l40s"

# Hardcoded wall times (auto-applied when 'original' present)
ORIGINAL_WALLTIME = "14:00:00"
OTHER_WALLTIME = "10:00:00"
DRYRUN_WALLTIME = "2:00:00"
SAND_WALLTIME = "24:00:00"

from high_tab.utils.io import abbrev_methods
from high_tab.utils.data_preparation import load_dataset

# -----------------------------------------------------------------------------
# Methods
# -----------------------------------------------------------------------------
ALLOWED_METHODS = (
    "original random_fs variance_fs tree_fs kbest_fs "
    "pca_dr random_dr kpca_dr agglo_dr kbest+pca sand_fs lasso_fs tabpfn_fs" 
).split()

# -----------------------------------------------------------------------------
# Slurm template (keeps CUDA-related exports unconditional to match behavior)
# -----------------------------------------------------------------------------
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
{gres_line}
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu=8GB
#SBATCH --export=ALL
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
{mail_line}{array_line}

# CUDA memory management / debugging
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TORCH_SHOW_CPP_STACKTRACES=1
export OMP_NUM_THREADS={cpus_per_task}
export MKL_NUM_THREADS={cpus_per_task}

{data_source_vars}
MODEL={model}

WORKING_DIR={working_dir}
VENV_DIR={venv_dir}
PYTHON_EXE={python_exe}
PYTHON_MODULE="high_tab.train"

methods=({methods_array})
METHOD=${{methods[$SLURM_ARRAY_TASK_ID]}}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PREFIX="${{METHOD}}_${{TIMESTAMP}}"
LOG_ID="${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"

# Ensure log/result dirs exist and redirect logs
exec >  "{log_dir}/${{LOG_PREFIX}}.${{LOG_ID}}.out" \
    2> "{log_dir}/${{LOG_PREFIX}}.${{LOG_ID}}.err"

echo "Running method ${{METHOD}} with ${{MODEL}} on {data_description}"

cd "$WORKING_DIR"

RESULTS_DIR="{results_root}/${{METHOD}}"
MODEL_CHECKPOINTS_DIR="${{RESULTS_DIR}}/model_checkpoints"
mkdir -p "$RESULTS_DIR"
mkdir -p "$MODEL_CHECKPOINTS_DIR"

# Sanity-check python binary
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "❌ python not found or not executable at $PYTHON_EXE" >&2
  exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Run python script
"$PYTHON_EXE" -m "$PYTHON_MODULE" \\
  --method "$METHOD" \\
  {data_source_arg} \\
  --model "$MODEL" \\
  --results_dir "$RESULTS_DIR" \\
  --model_checkpoints_dir "$MODEL_CHECKPOINTS_DIR" {save_time_flag}{dry_run_flag}{forward_args}


"""

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_methods(arg: str) -> List[str]:
    if arg.lower() in {"all", "*"}:
        return ALLOWED_METHODS.copy()
    lst = [m.strip() for m in arg.split() if m.strip()]
    bad = [m for m in lst if m not in ALLOWED_METHODS]
    if bad:
        raise ValueError(f"Unknown method(s): {', '.join(bad)}")
    return lst


def clean(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def safe_load_dataset(oid: int) -> str:
    obj = load_dataset(int(oid))
    return obj[1].name if isinstance(obj, tuple) else obj.name


def forward_passed_args(tokens: List[str]) -> str:
    return "" if not tokens else f" \\\n  {' '.join(tokens)}"


def generate_run_timestamp() -> str:
    # Default run name without seconds
    return datetime.now().strftime("%Y%m%d_%H%M")


def extra_args_suffix(tokens: List[str]) -> str:
    """
    Turn forwarded CLI tokens into a filename-safe suffix.
    Examples:
      ["--num_features", "123", "--fs_ratio=0.6", "--seed", "42", "--flag-only"]
      -> "_num_features_123_fs_ratio_0_6_seed_42_flag_only"
    """
    parts = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok.lstrip("-")
            if "=" in key:
                k, v = key.split("=", 1)
                parts.append(f"{clean(k)}_{clean(v)}")
            else:
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    parts.append(f"{clean(key)}_{clean(tokens[i + 1])}")
                    i += 1
                else:
                    parts.append(clean(key))
        elif tok.startswith("-"):
            parts.append(clean(tok.lstrip("-")))
        i += 1
    return "" if not parts else "_" + "_".join(parts)


def _read_flag_value(tokens: List[str], key: str):
    """Read --key=val or --key val from tokens. Returns string or None."""
    key_prefix = f"--{key}"
    for i, t in enumerate(tokens):
        if t == key_prefix and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            return tokens[i + 1]
        if t.startswith(key_prefix + "="):
            return t.split("=", 1)[1]
    return None


def parse_resource_overrides(tokens: List[str]):
    """
    Returns (num_gpus, num_cpus, device) where values may be None if not provided.
    device can be 'gpu' or 'cpu' if provided.
    """
    num_gpus = _read_flag_value(tokens, "num_gpus")
    num_cpus = _read_flag_value(tokens, "num_cpus")
    device = _read_flag_value(tokens, "device")

    num_gpus = int(num_gpus) if num_gpus is not None else None
    num_cpus = int(num_cpus) if num_cpus is not None else None
    device = device.lower() if isinstance(device, str) else None
    if device not in {None, "gpu", "cpu"}:
        device = None
    return num_gpus, num_cpus, device


def write_script(
    *,
    methods_list: List[str],
    time_str: str,
    args,
    logs_dir: Path,
    scripts_dir: Path,
    results_root: Path,
    dataset_name: str,
    mail_line: str,
    dry_run_flag: str,
    save_time_flag: str,
    forward_args: str,
    script_suffix: str,
    gres_line: str,
    cpus_per_task: int,
    data_source_vars: str,
    data_source_arg: str,
    data_description: str,
):
    abbr = abbrev_methods(methods_list)
    extra = "_test" if args.dry_run else ""
    script_base = f"{abbr}{extra}{script_suffix}"
    script_path = scripts_dir / f"{script_base}.sh"
    
    if len(methods_list) == 1:
        array_line = "#SBATCH --array=0"
    else:
        limit = "%4" if len(methods_list) > 4 else ""
        array_line = f"#SBATCH --array=0-{len(methods_list)-1}{limit}"
        
    slurm_text = SLURM_TEMPLATE.format(
        partition=args.partition,
        time=time_str,
        job_name=f"{args.model}_{dataset_name}",
        log_dir=logs_dir,
        script_base=script_base,
        mail_line=mail_line,
        array_line=array_line,
        working_dir=args.working_dir,
        venv_dir=args.venv_dir,
        python_exe=args.python_exe,
        methods_array=" ".join(methods_list),
        results_root=results_root,
        model=args.model,
        dry_run_flag=dry_run_flag,
        save_time_flag=save_time_flag,
        forward_args=forward_args,
        gres_line=gres_line,
        cpus_per_task=cpus_per_task,
        data_source_vars=data_source_vars,
        data_source_arg=data_source_arg,
        data_description=data_description,
    )

    with script_path.open("w") as f:
        f.write(dedent(slurm_text))
    os.chmod(script_path, 0o755)
    print("Submit with:\n  sbatch", script_path,"\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Slurm array scripts for TabPFN/OpenML runs.")
    
    # Make openml_id and csv_path mutually exclusive but require one
    data_group = ap.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--openml_id", type=str, help="OpenML task ID")
    data_group.add_argument("--csv_path", type=str, help="Path to CSV file")
    
    ap.add_argument("--methods", required=True, help="Space-separated list or 'all'")
    ap.add_argument("--exp_group", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--save_time", action="store_true")
    ap.add_argument("--run_id", help="Custom run ID (default: auto-generated timestamp)")

    # Cluster/env options
    ap.add_argument("--working_dir", default=WORK_DIR)
    ap.add_argument("--partition", default=DEF_PARTITION)
    ap.add_argument("--mail_user")
    ap.add_argument("--venv_dir", default=VENV_DIR)
    ap.add_argument("--python_exe", default=PYTHON_EXE)

    args, extra_args = ap.parse_known_args()

    # Methods
    methods = parse_methods(args.methods)
    
    # Determine dataset name and description
    if args.openml_id:
        try:
            dataset_raw = safe_load_dataset(args.openml_id)
            dataset_name = clean(dataset_raw)
        except Exception as e:
            sys.exit(f"❌ Could not load dataset: {e}")
        data_source_vars = f"OPENML_ID={args.openml_id}"
        data_source_arg = '--openml_id "$OPENML_ID"'
        data_description = f"OpenML task ${{OPENML_ID}}"
    else:  # csv_path
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            sys.exit(f"❌ CSV file not found: {args.csv_path}")
        dataset_name = clean(csv_path.stem)
        data_source_vars = f'CSV_PATH="{args.csv_path}"'
        data_source_arg = '--csv_path "$CSV_PATH"'
        data_description = f"CSV ${{CSV_PATH}}"

    # Run name
    run_id = args.run_id if args.run_id else generate_run_timestamp()
    print(f"Run ID: {run_id}")

    # Base dirs
    base = Path(args.working_dir, "experiments")
    logs_dir = base / "slurm_logs" / args.exp_group / run_id / dataset_name / args.model
    scripts_dir = base / "slurm_scripts" / args.exp_group / run_id / dataset_name / args.model
    results_root = base / "results" / args.exp_group / run_id / dataset_name / args.model

    for d in (logs_dir, scripts_dir, results_root):
        d.mkdir(parents=True, exist_ok=True)

    # Flags / header lines
    dry_run_flag = " \\\n  --dry_run" if args.dry_run else ""
    save_time_flag = " \\\n  --save_time" if args.save_time else ""
    mail_line = (
        f"#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={args.mail_user}\n" if args.mail_user else ""
    )
    forward_args = forward_passed_args(extra_args)
    suffix = extra_args_suffix(extra_args)

    # Resource overrides
    num_gpus, num_cpus, device = parse_resource_overrides(extra_args)
    default_gpus, default_cpus = 1, 4

    # Infer device if not explicitly set
    if device is None:
        if num_gpus is not None:
            device = "gpu" if num_gpus > 0 else "cpu"
        else:
            device = "gpu"  # keep existing default behavior

    final_cpus = num_cpus if num_cpus is not None else default_cpus
    if device == "gpu":
        final_gpus = num_gpus if num_gpus is not None else default_gpus
        gres_line = f"#SBATCH --gres=gpu:{final_gpus}"
    else:
        gres_line = ""

    # Walltimes
    if args.dry_run:
        time_original = DRYRUN_WALLTIME
        time_other = DRYRUN_WALLTIME
        time_sand = DRYRUN_WALLTIME
    else:
        time_original = ORIGINAL_WALLTIME
        time_other = OTHER_WALLTIME
        time_sand = SAND_WALLTIME

    # Prepare job groups: split out 'original' and 'sand_fs' into their own scripts
    jobs: List[tuple[List[str], str]] = []
    remaining = methods.copy()
    if "original" in remaining:
        jobs.append((["original"], time_original))
        remaining = [m for m in remaining if m != "original"]
    if "sand_fs" in remaining:
        jobs.append((["sand_fs"], time_sand))
        remaining = [m for m in remaining if m != "sand_fs"]
    if remaining:
        jobs.append((remaining, time_other))

    # Emit scripts
    for methods_list, t in jobs:
        write_script(
            methods_list=methods_list,
            time_str=t,
            args=args,
            logs_dir=logs_dir,
            scripts_dir=scripts_dir,
            results_root=results_root,
            dataset_name=dataset_name,
            mail_line=mail_line,
            dry_run_flag=dry_run_flag,
            save_time_flag=save_time_flag,
            forward_args=forward_args,
            script_suffix=suffix,
            gres_line=gres_line,
            cpus_per_task=final_cpus,
            data_source_vars=data_source_vars,
            data_source_arg=data_source_arg,
            data_description=data_description,
        )


if __name__ == "__main__":
    main()