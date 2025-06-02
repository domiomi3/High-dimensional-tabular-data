import argparse
import json
import logging
import openml 
import os
import warnings

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration


def setup_logger(log_level: str):
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


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


def run_tabpfn(X_train, y_train, X_test, y_test, ignore_pretraining_limits=False):
    # From the Prior-Labs repo
    from tabpfn import TabPFNRegressor

    # Initialize the regressor
    regressor = TabPFNRegressor(ignore_pretraining_limits=ignore_pretraining_limits)  
    regressor.fit(X_train, y_train)

    # Predict on the test set
    predictions = regressor.predict(X_test)

    # Evaluate the model
    # TODO: new sklearn allows rmse, see normalized rmse
    rmse = root_mean_squared_error(y_test, predictions)
    norm_rmse = rmse/np.std(y_test)
    return rmse, norm_rmse
    

def preprocess_dataset(X_train, X_test, y_train, metadata_flag, method, random_state, config, logger):
    # Load experiment config
    max_num_feat = config["max_num_feat"]
    num_tree_estimators = config["num_tree_estimators"]
    var_threshold = config["var_threshold"]

    if method == "original":
        if metadata_flag:
            logger.info("Category: Original")
            logger.info("Method: None")
            logger.info(f"Using original dataset with {X_train.shape[1]} features.")

    elif method == "random_fs":
        if metadata_flag:
            logger.info("Category: Feature Selection")
            logger.info(f"Method: Random selection")
            logger.info(f"HPs: Maximum number of features: {max_num_feat}")
        columns = X_train.sample(n=max_num_feat, axis=1, random_state=random_state).columns
        X_train = X_train[columns]
        X_test = X_test[columns]

    elif method == "variance_fs":
        if metadata_flag:
            logger.info("Category: Feature Selection")
            logger.info(f"Method: Variance threshold.")
            logger.info(f"HPs: Threshold: {var_threshold}")
        var_FS = VarianceThreshold(threshold=(var_threshold * (1 - var_threshold)))
        X_train = var_FS.fit_transform(X_train)
        X_test = var_FS.transform(X_test)

    elif method == "tree_fs":
        if metadata_flag:
            logger.info("Category: Feature Selection")
            logger.info(f"Method: Tree-based estimators")
            logger.info(f"HPs: Number of estimators: {num_tree_estimators}")
        clf = ExtraTreesRegressor(n_estimators=num_tree_estimators)
        clf = clf.fit(X_train, y_train)
        model = SelectFromModel(clf, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)

    elif method == "pca_dr":
        if metadata_flag:
            logger.info("Category: Dimensionality Reduction")
            logger.info(f"Method: PCA")
            logger.info(f"HPs: Maximum number of features: {max_num_feat}")
        pca = PCA(n_components=max_num_feat)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    elif method == "random_dr":
        if metadata_flag:
            logger.info("Category: Dimensionality Reduction")
            logger.info(f"Method: Gaussian random projection")
            logger.info(f"HPs: Maximum number of features: {max_num_feat}")
        transformer = GaussianRandomProjection(random_state=random_state, n_components=max_num_feat)
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)

    elif method == "agglo_dr":
        if metadata_flag:
            logger.info("Category: Dimensionality Reduction")
            logger.info(f"Method: Feature agglomeration")
            logger.info(f"HPs: Maximum number of clusters: {max_num_feat}")
        agglo = FeatureAgglomeration(n_clusters=max_num_feat)
        X_train = agglo.fit_transform(X_train)
        X_test = agglo.transform(X_test)

    else:
        logger.warning(f"{method} method unknown.")
    return X_train, X_test, y_train


def run_training(X, y, task, method, config, logger):
    # Load task config
    random_state = config["random_state"]

    # predefined splits from task
    if config["dry_run"]:
        n_repeats, n_folds, n_samples = (2,2,1)
        logger.info(f"Dry run mode with {n_repeats} repeats, {n_folds} folds, {n_samples} sample")
        logger.info("======================")
    else:
        n_repeats, n_folds, n_samples = task.get_split_dimensions()
        logger.info(f"Using split dimensions: repeats={n_repeats}, folds={n_folds}, samples={n_samples}")
        logger.info("======================")

    # for saving intemediate results
    dataset_name = config["dataset_name"]
    results_dir = config["results_dir"]
    timestamp = config["timestamp"]
    os.makedirs(results_dir, exist_ok=True)
    temp_results_path = f"{results_dir}/{dataset_name}_{method}_{timestamp}.csv"

    results_arr = [] # array for saving per step results + metadata
    metadata_flag = True # print metadata once

    # cross-validation
    for repeat_idx in range(n_repeats):
        for fold_idx in range(n_folds):
            fold_random_state = random_state + 100 * repeat_idx + fold_idx
         
            for sample_idx in range(n_samples):
                try:
                    # preprocess the dataset and run TabPFN 
                    train_indices, test_indices = task.get_train_test_split_indices(
                        repeat=repeat_idx,
                        fold=fold_idx,
                        sample=sample_idx,
                    )
                    X_train = X.iloc[train_indices]
                    y_train = y.iloc[train_indices]
                    X_test = X.iloc[test_indices]
                    y_test = y.iloc[test_indices]

                    # metada logging
                    if metadata_flag:
                        logger.info("")
                        logger.info("Dimensionality reduction:")
                        logger.info("======================")
                        #use DR method if applicable
                        X_train, X_test, y_train = preprocess_dataset(
                            X_train, X_test, y_train, metadata_flag, method, 
                            fold_random_state, config, logger
                        )
                        logger.info(
                            "Dataset size: X_train: %s, y_train %s, X_test %s, y_test %s",
                            X_train.shape, y_train.shape, X_test.shape, y_test.shape
                        )
                        if X_train.shape[1] > 1000:
                            logger.warning(
                                f"Number of features exceeding 1000! TabPFN will likely yield suboptimal predictions."
                            )
                        logger.info("======================")
                        logger.info("")
                        logger.info("Training:")
                        logger.info("======================")

                    # training step + saving and logging results
                    ignore_limits = True if X_train.shape[1] > 500 else False
                    rmse, norm_rmse = run_tabpfn(X_train, y_train, X_test, y_test, ignore_pretraining_limits=ignore_limits)
                    results_arr.append({
                        "dataset_name": config["dataset_name"],
                        "method": method,
                        "repeat": repeat_idx,
                        "fold": fold_idx,
                        "sample": sample_idx,
                        "rmse": rmse,
                        "norm_rmse": norm_rmse,
                        "n_features": X_train.shape[1]
                    })                    
                    logger.info(
                        f"Repeat #{repeat_idx}, fold #{fold_idx}, sample #{sample_idx}, seed: {fold_random_state} "
                        f"---------------- RMSE:{rmse:.4f}, normRMSE: {norm_rmse:.4f}"
                    )
                except Exception as e:
                    logger.error(
                        "Failure in repeat %d, fold %d, sample %d, method %s",
                        repeat_idx, fold_idx, sample_idx, method
                    )
                    logger.exception(e)  # Logs traceback
                finally:
                    metadata_flag = False  

    # save per-method results
    results_pd = pd.DataFrame(results_arr)
    results_pd.to_csv(temp_results_path, index=False)
    logger.info(f"Saved {method} results to {temp_results_path}")
    
    return results_arr


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
    logger.info(f"Loaded OpenML task: {task.task_id} - {task.task_type} on OpenML dataset {config['dataset_name']}")

    # Retrieve splits and data from the task
    X, y = task.get_X_and_y(dataset_format="dataframe")
    logger.info(f"Loaded dataset with shape X: {X.shape}, y: {y.shape}")

    # Run TabPFN with selected method(s)
    all_results = []
    for method in config["method"]:
        results_method = run_training(X, y, task, method, config, logger)
        all_results.extend(results_method)

    # Save results to csv
    results_dir = config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/{config["dataset_name"]}_all_{timestamp}.csv"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved all results to {results_path}")

    # Save config to json
    config_path = f"{results_dir}/{config['dataset_name']}_config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved config to {config_path}")

    # log summary
    summary = results_df.groupby("method")[["rmse", "norm_rmse"]].agg(["mean", "std"]).round(4)
    logger.info("Summary:\n%s", summary.to_string())

def main(config):
    # Set up logger
    logger = setup_logger(config["log_level"].upper())

    # Ignore TabPFN warnings about feature space dimension
    warnings.filterwarnings("ignore", message="Number of features .* is greater than the maximum.*")


    for dataset in config["dataset"]:
        dataset_config = config.copy()
        dataset_config["dataset"] = dataset # single dataset
        run_on_dataset(dataset_config, logger)
   


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run TabPFN with optional sklearn feature selection or dimensionality reduction methods.")
    parser.add_argument("--dataset", type=str, nargs="+", default=["363697"], help="One or more OpenML task IDs or dataset names.")
    parser.add_argument("--method", type=str, nargs="+", default=["original", "tree_fs"], choices=[
        "original", "random_fs", "variance_fs", "tree_fs", "pca_dr", "random_dr", "agglo_dr"
    ])
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--dry_run", action="store_true",
        help="Run a quick debug experiment with 2 repeats, 2 folds, 1 sample"
    )
    parser.add_argument("--random_state", type=int, default=44)
    parser.add_argument("--max_num_feat", type=int, default=500)
    parser.add_argument("--var_threshold", type=float, default=0.95)
    parser.add_argument("--num_tree_estimators", type=int, default=15)
    parser.add_argument("--log_level", type=str, default="INFO", choices=[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ])

    args = parser.parse_args()
    config = vars(args)

    main(config)