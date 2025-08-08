import logging
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def _yaml_safe(o):
    """Recursively convert non-serialisable objects (e.g. functions) to str."""
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, (list, tuple)):
        return [_yaml_safe(v) for v in o]
    if isinstance(o, dict):
        return {k: _yaml_safe(v) for k, v in o.items()}     
    return str(o)


def save_config_to_yaml(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(_yaml_safe(obj), f, sort_keys=False)
    logger.debug("Saved YAML to %s", path)


def save_results_to_csv(results_list, path):
    df = pd.DataFrame(results_list)
    df.to_csv(path, index=False)
    logger.info(f"Saved results to {path}")


def save_run(results, config, fold_times):
    """
    Save experiment config to yaml and per-fold run results to csv.
    """
    check_time = config["check_time"]
    if check_time:
        avg_fold = float(np.mean(fold_times)) if fold_times else None
        total_sec = float(sum(fold_times)) if fold_times else None
        config.update({
            "avg_fold_time": round(avg_fold, 4),
            "total_elapsed_time": round(total_sec, 4)
        })
        logger.info("Average fold time for '%s': %.1fs over %d folds",
                    config["method"], avg_fold, len(fold_times))

    out_dir = Path(config["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / config["results_file"]
    save_results_to_csv(results, csv_path)

    yaml_path = out_dir / config["config_file"]
    save_config_to_yaml({
        **config
    }, yaml_path)


def extract_hyperparams_from_filename(path):
    filename = os.path.basename(path)
    pattern = r"_s(?P<seed>\d+)_nf(?P<nf>\d+)_vt(?P<vt>[\d.]+)_nte(?P<nte>\d+)"
    match = re.search(pattern, filename)
    if not match:
        return {}
    return {
        "seed": int(match.group("seed")),
        "nf": int(match.group("nf")),
        "vt": float(match.group("vt")),
        "nte": int(match.group("nte"))
    }


def rename_run_file(config):
    id = config["dataset_name"] + "_" + config["method"] + "_" + config["time_start"]
    config["config_file"] = id + "_config.yaml"
    config["results_file"] = id + "_results.csv"

