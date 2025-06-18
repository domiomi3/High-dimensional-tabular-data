import argparse
import gc
import openml 
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.metrics import root_mean_squared_error, roc_auc_score, log_loss

from preprocess_sklearn import *
sys.path.append(".")
from utils import *


def load_dataset(dataset_id_or_name):
    try:
        task_id = int(dataset_id_or_name)
        task = openml.tasks.get_task(task_id)
    except ValueError:
        dataset_list = openml.datasets.list_datasets(output_format="dataframe")
        matching = dataset_list[dataset_list['name'] == dataset_id_or_name]
        if matching.empty:
            raise ValueError(f"No OpenML dataset found with name '{dataset_id_or_name}'")
        dataset_id = matching.iloc[0]['did']
        task_list = openml.tasks.list_tasks(output_format="dataframe", dataset=dataset_id)
        if task_list.empty:
            raise ValueError(f"No tasks found for dataset '{dataset_id_or_name}'")
        task_id = task_list.iloc[0]['tid'] # first matching task
        task = openml.tasks.get_task(task_id)

    dataset = task.get_dataset()
    return task, dataset


def infer_tabpfn(X_train, y_train, X_test, y_test, task_type, ignore_pretraining_limits=False, metric_override=None):
    from tabpfn import TabPFNRegressor, TabPFNClassifier

    if task_type == "regression":
        model = TabPFNRegressor(ignore_pretraining_limits=ignore_pretraining_limits)  
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_cleanup(model)
        return root_mean_squared_error(y_test, predictions)

    elif task_type == "classification":
        model = TabPFNClassifier(ignore_pretraining_limits=ignore_pretraining_limits)
        model.fit(X_train, y_train)
        if metric_override == "log_loss":
            probs = model.predict_proba(X_test)
            model_cleanup(model)
            return log_loss(y_test, probs)
        else:  # default to roc auc
            preds = model.predict(X_test)
            model_cleanup(model)
            if len(y_train.unique()) > 2: #multi-class
                return roc_auc_score(y_test, preds, multi_class='ovr')
            else:
                return roc_auc_score(y_test, preds)

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def train(X, y, task, method, config, logger):
    # Load task config
    n_repeats = config["n_repeats"]
    n_folds = config["n_folds"]
    n_samples = config["n_samples"]
    random_state = config["random_state"]
    task_type = config["task_type"]
    metric_override = config["metric_override"]

    results_list = [] # list for saving per step results + metadata
    metadata_flag = True # print metadata once
    
    # start logging 
    logger.info("")
    logger.info("Preprocessing:")
    logger.info("======================")

    # cross-validation
    for repeat_idx in range(n_repeats):
        for fold_idx in range(n_folds):
            fold_random_state = random_state + 100 * repeat_idx + fold_idx

            for sample_idx in range(n_samples):
                try: # preprocess the dataset and run TabPFN 
                    train_indices, test_indices = task.get_train_test_split_indices(
                        repeat=repeat_idx,
                        fold=fold_idx,
                        sample=sample_idx,
                    )
                    X_train = X.iloc[train_indices]
                    y_train = y.iloc[train_indices]
                    X_test = X.iloc[test_indices]
                    y_test = y.iloc[test_indices]

                    # metadata logging
                    if metadata_flag:
                        X_train, X_test, y_train, y_test = run_preprocessing_pipeline(
                            X_train, X_test, y_train, y_test, method, task_type,
                            random_state, config, logger, metadata_flag
                        )

                        logger.info(
                            "Dataset size: X_train: %s, y_train %s, X_test %s, y_test %s",
                            X_train.shape, y_train.shape, X_test.shape, y_test.shape
                        )
                        if X_train.shape[1] > 500:
                            logger.warning(
                                f"Number of features exceeding 500! TabPFN will likely yield suboptimal predictions."
                            )
                        logger.info("======================")
                        logger.info("")
                        logger.info("Training:")
                        logger.info("======================")
                    else: # no logging
                        X_train, X_test, y_train, y_test = run_preprocessing_pipeline(
                            X_train, X_test, y_train, y_test, method, task_type,
                            random_state, config, logger, metadata_flag
                        )
                    # training step + saving and logging results
                    ignore_limits = True if X_train.shape[1] > 500 else False
                    results_metric = infer_tabpfn(
                        X_train, y_train, X_test, y_test, task_type, 
                        ignore_pretraining_limits=ignore_limits, metric_override=metric_override
                    ) 
                                          
                    # include all task types
                    if task_type == "regression":
                        metric_name = "rmse"
                        extra_metrics = {"norm_rmse": results_metric / np.std(y_test)}
                    elif task_type == "classification":
                        metric_name = "log_loss" if config.get("metric_override") == "log_loss" else "auc_roc"
                        extra_metrics = {}
                    else:
                        raise ValueError(f"Unknown task type: {task_type}")

                    result_entry = {
                        "dataset_name": config["dataset_name"],
                        "method": method,
                        "repeat": repeat_idx,
                        "fold": fold_idx,
                        "sample": sample_idx,
                        "n_features": X_train.shape[1],
                        metric_name: results_metric,
                        **extra_metrics
                    }
                    results_list.append(result_entry)

                    # format log message
                    log_metric_name = {
                        "rmse": "RMSE",
                        "norm_rmse": "normRMSE",
                        "auc_roc": "AUC ROC",
                        "log_loss": "Log Loss"
                    }[metric_name]
                    logger.info(
                        f"Repeat #{repeat_idx}, fold #{fold_idx}, sample #{sample_idx}, seed: {fold_random_state} "
                        f"---------------- {log_metric_name}:{results_metric:.4f}"
                        + (f", normRMSE: {extra_metrics['norm_rmse']:.4f}" if "norm_rmse" in extra_metrics else "")
                    )

                    # memory clean-up
                    del X_train, X_test, y_train, y_test
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(
                        "Failure in repeat %d, fold %d, sample %d, method %s",
                        repeat_idx, fold_idx, sample_idx, method
                    )
                    logger.exception(e)  # Logs traceback
                finally:
                    metadata_flag = False  

    # save per-method results
    os.makedirs(config["results_dir"], exist_ok=True)
    temp_results_path = f'{generate_filename_prefix(config, config["method"][0])}.csv'
    save_results_to_csv(results_list, temp_results_path, logger)

    # Save per-method config to json
    config_path = f'{generate_filename_prefix(config, config["method"][0])}_config.json'
    save_config_to_json(config, config_path, logger)
    
    return results_list


def run_on_dataset(config, logger):
    # timestamp for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.update({
        "timestamp": timestamp
    })

    # Load OpenML dataset and task
    task, dataset = load_dataset(config["dataset"])
    
    config.update({
        "dataset_name": dataset.name,
    })
    logger.info("")
    logger.info("Dataset:")
    logger.info("======================")
    logger.info(f"OpenML task: {task.task_id}")
    logger.info(f"Task type: {task.task_type}")
    logger.info(f"OpenML dataset: {config['dataset_name']}")

    # Retrieve splits and data from the task
    X, y = task.get_X_and_y(dataset_format="dataframe")
    logger.info(f"Dataset size: X: {X.shape}, y: {y.shape}")

    if config["dry_run"]:
        n_repeats, n_folds, n_samples = (2,2,1)
    else:     # predefined splits from task
        n_repeats, n_folds, n_samples = task.get_split_dimensions()
    config.update({
        "n_repeats": n_repeats,
        "n_folds": n_folds,
        "n_samples": n_samples
        })

    logger.info(f"Split dimensions: repeats={n_repeats}, folds={n_folds}, samples={n_samples}")
    logger.info("======================")

    # Retrieve task type
    task_type_str = task.task_type  # Replace with your actual object or string source
    task_type = None
    if re.search(r"regression", task_type_str, re.IGNORECASE):
        task_type = "regression"
    elif re.search(r"classification", task_type_str, re.IGNORECASE):
        task_type = "classification"
    else:
        raise ValueError(f"Unknown task type: {task_type_str}")
    config.update({"task_type": task_type})

    # Run TabPFN with selected method(s)
    all_results = []
    for method in config["method"]:
        results_method = train(X, y, task, method, config, logger)
        all_results.extend(results_method)
    results_dir = config["results_dir"]

    # Save all results to csv
    if len(config["method"]) > 1: # results for a single method are already saved in run_training
        os.makedirs(results_dir, exist_ok=True)
        results_path = f'{generate_filename_prefix(config, "all")}.csv'
        save_results_to_csv(all_results, results_path, logger)

        # Save config to json
        config_path = f'{generate_filename_prefix(config, "all")}_config.json'
        save_config_to_json(config, config_path, logger)

    # log summary
    results_df = pd.DataFrame(all_results)
    if task_type == "regression":
        summary = results_df.groupby("method")[["rmse", "norm_rmse"]].agg(["mean", "std"]).round(4)
    elif task_type == "classification":
        if config.get("metric_override") == "log_loss":
            summary = results_df.groupby("method")["log_loss"].agg(["mean", "std"]).round(4)
        else:
            summary = results_df.groupby("method")["auc_roc"].agg(["mean", "std"]).round(4)
    logger.info("Summary:\n%s", summary.to_string())


def main(config):
    # Set up logger
    logger = setup_logger(config["log_level"].upper())

    # Ignore TabPFN warnings about feature space dimension
    warnings.filterwarnings("ignore", message="Number of features .* is greater than the maximum.*")

    # iterate over specified datasets and train tabpfn
    for dataset in config["dataset"]:
        dataset_config = config.copy()
        dataset_config["dataset"] = dataset # single dataset
        run_on_dataset(dataset_config, logger)
   

if __name__=="__main__":
    ALL_METHODS = [
        "original", "random_fs", "variance_fs", "tree_fs", "kbest_fs",
        "pca_dr", "random_dr", "agglo_dr", "ica_dr", "kpca_dr"
    ]

    parser = argparse.ArgumentParser(description="Run TabPFN with optional sklearn feature selection or dimensionality reduction methods.")
    parser.add_argument("--dataset", type=str, nargs="+", default=["363697"], help="One or more OpenML task IDs or dataset names.")
    parser.add_argument(""
        "--method", type=str, nargs="+", default=["original", "tree_fs"],
        help="Feature selection or dimensionality reduction methods. Use 'all' to run all available methods."
    )
    parser.add_argument(
        "--metric_override", type=str, default=None,
        help="Override the default metric (e.g., 'log_loss' for specific classification tasks)"
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--dry_run", action="store_true",
        help="Run a quick debug experiment with 2 repeats, 2 folds, 1 sample"
    )
    parser.add_argument("--random_state", type=int, default=44)
    parser.add_argument("--n_features", type=int, default=500)
    parser.add_argument("--var_threshold", type=float, default=0.95)
    parser.add_argument("--n_tree_estimators", type=int, default=15)
    parser.add_argument("--log_level", type=str, default="INFO", choices=[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ])

    args = parser.parse_args()
    config = vars(args)

    if "all" in config["method"]:
        config["method"] = ALL_METHODS

    main(config)