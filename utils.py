import os
import re
import gc
import logging
import json
import psutil
import pandas as pd
import torch


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
    with open(path, "w") as f:
            json.dump(config, f, indent=4) 
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
