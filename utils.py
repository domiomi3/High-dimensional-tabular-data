import os
import logging
import json
import pandas as pd

def generate_filename_prefix(config, method="all"):
    base = (
        f"{config['dataset_name']}_{method}"
        f"_s{config['random_state']}"
        f"_nf{config['max_num_feat']}"
        f"_vt{config['var_threshold']}"
        f"_nte{config['num_tree_estimators']}"
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

