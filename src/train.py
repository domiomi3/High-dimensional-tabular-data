#!/usr/bin/env python3

import argparse
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from preprocess import run_preprocessing_pipeline    
from utils.openml_data import prepare_data
from utils.io import save_run, rename_run_file   
from utils.hardware import get_device, memory_cleanup, ensure_gpu_or_die
from utils.loggers import setup_logger                  
from models import make_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
#  Training loop                                                        #
# ---------------------------------------------------------------------------#
def train(X, y, task, config) -> List[dict]:
    # read config
    n_rep, n_fold, n_samp = config["n_repeats"], config["n_folds"], config["n_samples"]
    task_type   = config["task_type"]
    eval_metric = config["eval_metric"]
    eval_func = config["eval_func"]
    ignore_limits = config["ignore_limits"]
    model_key   = config["model"]
    check_time  = config["check_time"]

    device = get_device()
    results: List[dict] = []
    fold_times: List[float] = []
    first_iter_flag = True

    logger.info("")            
    logger.info("Preprocessing:")
    logger.info("======================")

    # training loop
    for rep in range(n_rep):
        rep_seed = config["random_state"] + 100 * rep # random seed per repeat
        for fold in range(n_fold):
            for samp in range(n_samp):
                if check_time:
                    t0 = time.time()

                try:
                    # get OpenML task train/test splits
                    train_idx, test_idx = task.get_train_test_split_indices(
                        repeat=rep, fold=fold, sample=samp)
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
                    

                    # preprocess data
                    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(
                        X_train, X_test, y_train, y_test,
                        rep_seed, config, first_iter_flag)
                    
                    # instantiate model
                    if model_key == "tabpfnv2_org":
                        extra_args = {"ignore_limits": ignore_limits}
                    else:
                        extra_args = {}
                    model = make_model(model_key,
                        task_type=task_type,
                        eval_metric=eval_metric,
                        eval_func=eval_func,
                        device=device,
                        **extra_args
                    )

                    # logging metadata in the first iteration
                    if first_iter_flag:
                        logger.info(
                            "Dataset size: X_train %s  y_train %s  "
                            "X_test %s  y_test %s",
                            X_train.shape, y_train.shape, X_test.shape, y_test.shape
                        )
                        if X_train.shape[1] > 500:
                            logger.warning("Number of features >500, "
                                           "TabPFN may underperform.")
                        logger.info("") 
                        logger.info("Training")
                        logger.info("======================")
                        logger.info(f"Model: {model.model_name} implemented with {model.__class__.__name__} class")
                        logger.info("") 
                        first_iter_flag = False

                    # train model
                    model = model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    memory_cleanup(model, X_train, X_test, y_train, y_test)

                    # add norm_rmse
                    extra = {}
                    if task_type == "regression" and "norm_rmse" in eval_metric:
                        extra["norm_rmse"] = score / np.std(y_test)
                    if check_time:
                        fold_time = time.time() - t0
                        fold_times.append(fold_time)
                        extra["fold_time"]=round(fold_time, 4)

                    # save per-sample (fold) results
                    results.append({
                        "dataset_name": config["dataset_name"],
                        "model": config["model"],
                        "method": config["method"],
                        "repeat": rep,
                        "fold": fold,
                        "sample": samp,
                        eval_metric[0]: round(score,4),
                        **extra,
                    })
                    logger.info(
                        "repeat %d | fold %d | sample %d | %s: %.4f%s",
                        rep, fold, samp, eval_metric[0], score,
                        f" | normRMSE: {extra['norm_rmse']:.4f}" if 'norm_rmse' in extra else ""
                    )

                except Exception as e:
                    logger.error("Failure r%02d f%02d s%02d m%-10s",
                                 rep, fold, samp, config["method"])
                    logger.exception(e)


    return results, fold_times


# ---------------------------------------------------------------------------#
#  Main                                                                       #
# ---------------------------------------------------------------------------#
def main(config):

    # setup
    setup_logger(config["log_level"].upper())
    ensure_gpu_or_die()

    warnings.filterwarnings("ignore",
        message="Number of features .* is greater than the maximum.*")
    warnings.filterwarnings("ignore",
        message="pkg_resources is deprecated as an API.*", category=UserWarning)
    warnings.filterwarnings("ignore",
        message="X does not have valid feature names.*", category=UserWarning)

    # get openml data
    X, y, task = prepare_data(config)
    Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)

    # train model for each method
    all_results: List[dict] = []
    for method in config["method"]:
        m_config = config.copy()
        m_config["method"] = method
        m_config["time_start"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        results, fold_times = train(X, y, task, m_config)
        m_config["time_finish"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        all_results.extend(results)
        if m_config["unique_save"]:
            rename_run_file(m_config)
        save_run(results, m_config, fold_times)

    # log performance summary for all methods
    df = pd.DataFrame(all_results)
    summary = df.groupby("method")[config["eval_metric"]].agg(["mean", "std"]).round(4)
    logger.info("\nSummary:\n%s", summary.to_string())


# ---------------------------------------------------------------------------#
#  Entry-point                                                                #
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    ALL_METHODS = [
        "original", "random_fs", "variance_fs", "tree_fs", "kbest_fs",
        "pca_dr", "random_dr", "agglo_dr", "kpca_dr",
    ]

    TABARENA_MODELS = {"tabpfnv2_tab": "TabPFNv2", "catboost_tab": "CatBoost", "realmlp_tab": "RealMLP"}

    parser = argparse.ArgumentParser(description="Run TabPFNv2 / CatBoost + FS/DR pipeline")
    parser.add_argument("--method", nargs="+", default=["original", "tree_fs"],
                    choices = ALL_METHODS, help="'all' to run every method.")
    parser.add_argument("--openml_id", type=str, default="363697")
    parser.add_argument("--model", default="tabpfnv2_org",
                  choices=["tabpfnv2_org", "tabpfnv2_tab", "catboost_tab", "realmlp_tab"])
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--results_file", default="results.csv")
    parser.add_argument("--config_file", default="config.yaml")
    parser.add_argument("--unique_save", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--check_time", action="store_true")
    parser.add_argument("--random_state", type=int, default=44)
    parser.add_argument("--n_features", type=int, default=500)
    parser.add_argument("--var_threshold", type=float, default=0.95)
    parser.add_argument("--n_tree_estimators", type=int, default=15)
    parser.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = parser.parse_args()
    config = vars(args)

    if "all" in config["method"]:
        config["method"] = ALL_METHODS

    main(config)
