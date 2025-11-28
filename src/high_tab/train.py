#!/usr/bin/env python3

import argparse
import logging
import time
import torch
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from high_tab.preprocess import run_preprocessing_pipeline    
from high_tab.utils.data_preparation import prepare_data
from high_tab.utils.misc import add_method_args
from high_tab.utils.io import save_run, rename_run_file   
from high_tab.utils.hardware import get_device, memory_cleanup, set_hardware_config, set_seed
from high_tab.utils.loggers import setup_logger                  
from high_tab.models import make_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
#  Training loop                                                        #
# ---------------------------------------------------------------------------#
def train(X, y, task, config) -> List[dict]:
    # read config
    num_rep, num_fold = config["num_repeats"], config["num_folds"]
    task_type   = config["task_type"]
    eval_metric = config["eval_metric"]
    ignore_limits = config["ignore_limits"]
    model_key = config["model"]
    preprocessing = config["preprocessing"]
    save_time  = config["save_time"]
    model_checkpoints_dir = config["model_checkpoints_dir"]
    device = config["device"] if config["device"] else get_device() 
    num_gpus = config["num_gpus"]
    num_cpus = config["num_cpus"]
    num_k_folds = config["num_k_folds"]
    test_df_idx = config["test_df_idx"]

    results: List[dict] = []
    fold_times: List[float] = []
    first_iter_flag = True

    logger.info("")            
    logger.info("Preprocessing:")
    logger.info("======================")

    # training loop
    for rep in range(num_rep):
        # seed = 0
        rep_seed = config["seed"] + 100 * rep # random seed per repeat
        for fold in range(num_fold):
            if save_time:
                t0 = time.time()

            try:
                # get OpenML task train/test splits
                if task:
                    train_idx, test_idx = task.get_train_test_split_indices(
                        repeat=rep, fold=fold)
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
                else:
                    fold_seed = rep_seed + fold

                    # tODO: split based on repreat and fold
                      # Split based on repeat and fold with deterministic seed
                    if test_df_idx is not None:
                        X_train, y_train = X.iloc[:test_df_idx], y.iloc[:test_df_idx]
                        X_test, y_test = X.iloc[test_df_idx:], y.iloc[test_df_idx:]
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, 
                            test_size=0.3,  # adjust as needed
                            random_state=fold_seed,
                            stratify=y if task_type in ("binary", "multiclass") else None  # stratify for classification
                        )
                    
                # preprocess data (default preprocessing + optional FS/DR)
                X_train, X_test, y_train, y_test = run_preprocessing_pipeline(
                    X_train, X_test, y_train, y_test,
                    rep_seed, config, first_iter_flag
                )
                
                # instantiate model
                extra_args = {}
                if model_key == "tabpfnv2_org":
                    extra_args["ignore_limits"] = ignore_limits
                extra_args.update(add_method_args(config))
                model_eval_metric = "root_mean_squared_error" if eval_metric[0]=="rmse" else eval_metric[0]
                model = make_model(
                    model_key=model_key,
                    model_checkpoints_dir=model_checkpoints_dir,
                    preprocessing=preprocessing,
                    task_type=task_type,
                    eval_metric=model_eval_metric,
                    random_state=rep_seed,
                    device=device,
                    num_gpus=num_gpus,
                    num_cpus=num_cpus,
                    num_k_folds=num_k_folds,
                    **extra_args
                )

                # logging metadata in the first iteration
                if first_iter_flag:
                    logger.info(
                        "Dataset size: X_train %s  y_train %s  "
                        "X_test %s  y_test %s",
                        X_train.shape, y_train.shape, X_test.shape, y_test.shape
                    )
                    
                    model_name = "TabPFN-Wide" if model_key == "tabpfn_wide" else model.model_name 
                    logger.info("") 
                    logger.info("Training")
                    logger.info("======================")
                    logger.info("Splits: repeats=%d folds=%d", config["num_repeats"], config["num_folds"])
                    logger.info(f"Model: {model_name} implemented with {model.__class__.__name__} class")
                    logger.info("") 
                    first_iter_flag = False

                # train model
                model = model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                # add norm_rmse and validation score
                extra = {}
                if task_type == "regression" and "norm_rmse" in eval_metric:
                    extra["norm_rmse"] = score / np.std(y_test)
                if save_time:
                    fold_time = time.time() - t0
                    fold_times.append(fold_time)
                    extra["fold_time"] = fold_time
                if model.val_score is not None:
                    val_score = model.val_score
                    extra[f"val_{eval_metric[0]}"] = val_score

                # save per-fold results
                results.append({
                    "dataset_name": config["dataset_name"],
                    "model": config["model"],
                    "method": config["method"],
                    "repeat": rep,
                    "fold": fold,
                    "seed": rep_seed,
                    eval_metric[0]: score,
                    **extra,
                })
                log_parts = [f"repeat {rep} | fold {fold} | seed {rep_seed}",
                        f"{eval_metric[0]}: {score:.4f}"]
                if f"val_{eval_metric[0]}" in extra:
                    log_parts.append(f"val_{eval_metric[0]}: {extra[f'val_{eval_metric[0]}']:.4f}")
                if 'norm_rmse' in extra:
                    log_parts.append(f"normRMSE: {extra['norm_rmse']:.4f}")
                logger.info(" | ".join(log_parts))

                # clean up memory
                del model, X_train, X_test, y_train, y_test
                memory_cleanup()

            except Exception as e:
                logger.exception(
                    "Failure r%02d f%02d s%02d m%-10s",
                    int(rep), int(fold), int(rep_seed), str(config["method"])
                )
                logger.exception(e)


    return results, fold_times


# ---------------------------------------------------------------------------#
#  Main                                                                       #
# ---------------------------------------------------------------------------#
def main(config):

    # setup
    set_seed(config["seed"])
    setup_logger(config["log_level"].upper())
    set_hardware_config(config)

    warnings.filterwarnings("ignore",
        message="Number of features .* is greater than the maximum.*")
    warnings.filterwarnings("ignore",
        message="pkg_resources is deprecated as an API.*", category=UserWarning)
    warnings.filterwarnings("ignore",
        message="X does not have valid feature names.*", category=UserWarning)

    # get openml data
    X, y, task, test_df_idx = prepare_data(config)
    config["test_df_idx"] = test_df_idx
    Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)

    # train model with each method
    all_results: List[dict] = []
    for method in config["method"]:
        m_config = config.copy()
        m_config["method"] = method
        m_config["num_kbest_features"] = round(m_config["fs_ratio"]*m_config["num_features"])
        m_config["num_pca_comps"] = m_config["num_features"]-m_config["num_kbest_features"]
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
        "pca_dr", "random_dr", "agglo_dr", "kpca_dr", "kbest+pca",
        "sand_fs", "lasso_fs", "tabpfn_fs"
    ]

    TABARENA_MODELS = {
        "tabpfnv2_ag": "TabPFNv2",
        "realtabpfn_tab": "TabPFN-v2.5", 
        "catboost_tab": "CatBoost", 
    }

    parser = argparse.ArgumentParser(
        description="Run TabArena model on high dimensional datasets from " \
        "TabArena benchmark with FS/DR preprocessing.")
    parser.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    # experiment type
    parser.add_argument("--method", nargs="+", default=["random_fs"],
                    choices = ALL_METHODS, help="Sklearn method/combination of methods.")
    parser.add_argument("--openml_id", type=str, default=None,
                    help="OpenML task ID or dataset name.")
    parser.add_argument("--csv_path", type=str, default=None,
                   help="Path to .csv with (training) data.")
    parser.add_argument("--test_csv_path", type=str, default=None,
                   help="Path to .csv with test data.")
    parser.add_argument("--target", type=str, default=None,
                   help="Target column namefor the .csv data.")
    parser.add_argument("--model", default="tabpfnv2_ag",
                  choices=["tabpfnv2_org", "tabpfn_wide", "tabpfnv2_ag", "catboost_tab", "realtabpfn_tab"],
                  help="TabArena model.")
    parser.add_argument("--preprocessing", default="model-specific", 
                        choices=["model-specific", "model-agnostic"], 
                        help="Whether to apply pre-split or per-fold preprocessing.")
    parser.add_argument("--num_repeats", type=int, help="Number of training repeats.")
    parser.add_argument("--num_folds", type=int, help="Number of training folds.")
    # experiment meta
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_gpus", type=int,)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--save_time", action="store_true",
                        help="Saves per-fold and elapsed times.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Testing run, uses 1 fold and 1 repeat.")
    # saving
    parser.add_argument("--model_checkpoints_dir", default="model_checkpoint/")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--results_file", default="results.csv")
    parser.add_argument("--config_file", default="config.yaml")
    parser.add_argument("--unique_save", 
                        action="store_true", 
                        help="Whether to save experiment results under unique filenames.")
    # method and model HPs
    parser.add_argument("--num_estimators", type=int, default=8,
                        help="Number of TabPFN estimators.")
    parser.add_argument("--num_k_folds", type=int, default=8,
                        help="Number of k-folds for bagged ensembling.")
    parser.add_argument("--num_features", type=int, default=150,
                        help="Resulting feature dimension after applying the sklearn method.")
    parser.add_argument("--fs_ratio", type=float, default=0.75,
                        help="The ratio between FS and DR features for combined methods.")
    parser.add_argument("--var_threshold", type=float, default=0.95,
                        help="For the VarianceThreshold method; remove features below that number.")
    parser.add_argument("--num_tree_estimators", type=int, default=10,
                        help="For the SelectFromModel method; number of randomized decision trees.")

    args = parser.parse_args()

    # data source
    if args.csv_path and args.openml_id:
        logger.warning("--csv_path and --openml_id both provided; using CSV and ignoring OpenML.")
        args.openml_id = None
    if not args.csv_path and not args.openml_id:
        args.openml_id = "363620"
    
    config = vars(args)
    if "all" in config["method"]:
        config["method"] = ALL_METHODS

    main(config)
