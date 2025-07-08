import os
import re
import gc
import logging
import json
import psutil
import pandas as pd
import torch
import yaml  
import openml 
import sys

from pathlib import Path

from sklearn.metrics import root_mean_squared_error, roc_auc_score, log_loss


def _yaml_safe(o):
    """Recursively convert non-serialisable objects (e.g. functions) to str."""
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, (list, tuple)):
        return [_yaml_safe(v) for v in o]
    if isinstance(o, dict):
        return {k: _yaml_safe(v) for k, v in o.items()}     
    return str(o)


def ensure_gpu_or_die(logger):
    if not torch.cuda.is_available():
        logger.error(
            "🚨  No CUDA-capable GPU detected -- aborting run. "
            "Set up a GPU or remove this safety check."
        )
        sys.exit(1)          # non-zero exit code → SBATCH job marked FAILED
    else:
        log_device_info(logger)


def log_device_info(logger):
    dev_id   = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(dev_id)
    logger.info("") 
    logger.info("Hardware configuration:")
    logger.info("======================")
    logger.info("Device: GPU %d – %s", dev_id, dev_name)
    logger.info("CUDA runtime: %s | PyTorch CUDA build: %s",
                torch.version.cuda, torch.version.git_version[:7])
    logger.info("GPUs visible: %d", torch.cuda.device_count())
    

def save_yaml(obj: dict, path: Path, logger):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(_yaml_safe(obj), f, sort_keys=False)
    logger.debug("Saved YAML to %s", path)


def generate_filename_prefix(config, method="all"):
    base = (
        f"{config['dataset_name']}_{method}"
        f"_s{config['random_state']}"
        f"_nf{config['n_features']}"
        f"_vt{config['var_threshold']}"
        f"_nte{config['n_tree_estimators']}"
        f"_{config['timestamp']}"
    )
    return os.path.join(config['results_dir'], base)


def save_results_to_csv(results_list, path, logger):
    df = pd.DataFrame(results_list)
    df.to_csv(path, index=False)
    logger.info(f"Saved results to {path}")


def save_config_to_json(config, path, logger):
    # convert non-serializable entries (e.g., functions) to strings
    serializable_config = {}
    for key, value in config.items():
        if callable(value):
            serializable_config[key] = value.__name__
        else:
            serializable_config[key] = value
    with open(path, "w") as f:
        json.dump(serializable_config, f, indent=4)
    logger.info(f"Saved config to {path}")



def setup_logger(log_level: str):
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def print_mem(logger):
    logger.info(f"CPU Memory Used: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"Total GPU memory: {total_mem / 1024**2:.2f} MB")
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def model_cleanup(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def ensure_dataframe(X, reference=None):
    """Ensure X is a pandas DataFrame. Optionally use column names from reference."""
    if isinstance(X, pd.DataFrame):
        return X
    if reference is not None and isinstance(reference, pd.DataFrame):
        return pd.DataFrame(X, columns=reference.columns)
    return pd.DataFrame(X)


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



def load_dataset(openml_id):
    try:
        task_id = int(openml_id)
        task = openml.tasks.get_task(task_id)
    except ValueError:
        dataset_list = openml.datasets.list_datasets(output_format="dataframe")
        matching = dataset_list[dataset_list['name'] == openml_id]
        if matching.empty:
            raise ValueError(f"No OpenML dataset found with name '{openml_id}'")
        dataset_id = matching.iloc[0]['did']
        task_list = openml.tasks.list_tasks(output_format="dataframe", dataset=dataset_id)
        if task_list.empty:
            raise ValueError(f"No tasks found for dataset '{openml_id}'")
        task_id = task_list.iloc[0]['tid'] # first matching task
        task = openml.tasks.get_task(task_id)

    dataset = task.get_dataset()
    return task, dataset


def get_task_type(task):
    # Retrieve task type
    task_type_str = task.task_type  # Replace with your actual object or string source
    task_type = None
    if re.search(r"regression", task_type_str, re.IGNORECASE):
        task_type = "regression"
        task_type_log = "Regression"
    elif re.search(r"classification", task_type_str, re.IGNORECASE):
        num_classes = len(task.class_labels)
        if num_classes > 2:
            task_type = "multiclass"
            task_type_log = f"Multiclass classification with {num_classes} classes"
        else:
            task_type = "binary"
            task_type_log = "Binary classification"
    else:
        raise ValueError(f"Unknown task type: {task_type_str}")
    return task_type, task_type_log


def get_eval_metric(task_type):
    if task_type == "regression": 
        return ["rmse", "norm_rmse"], root_mean_squared_error
    elif task_type == "binary":
        return ["roc_auc"], roc_auc_score
    elif task_type == "multiclass":
        return ["log_loss"], log_loss
    else:
        raise ValueError(f"No evaluation metric found for task type: {task_type}")

