import argparse
import json
import logging
import openml 
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration


def run_tabpfn(X_train, y_train, X_test, y_test, ignore_pretraining_limits=False):
    # From the Prior-Labs repo
    from tabpfn import TabPFNRegressor

    # Initialize the regressor
    regressor = TabPFNRegressor(ignore_pretraining_limits=ignore_pretraining_limits)  
    regressor.fit(X_train, y_train)

    # Predict on the test set
    predictions = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    
    return np.sqrt(mse)
    

def preprocess_dataset(X_train, X_test, y_train, metadata_flag, method, random_state, config):
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
        logger.warning(f"{method} method unkown.")
    return X_train, X_test, y_train


def run_training(X, y, method, config):
    # Load task config
    n_repeats = config["n_repeats"]
    n_folds = config["n_folds"]
    n_samples = config["n_samples"]
    random_state = config["random_state"]

    # for saving intemediate results
    dataset_name = config["dataset_name"]
    os.makedirs("results", exist_ok=True)
    temp_results_path = f"results/{dataset_name}_{method}.csv"

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
                            X_train, X_test, y_train,
                            metadata_flag, method, fold_random_state, config
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
                    rmse = run_tabpfn(X_train, y_train, X_test, y_test, ignore_pretraining_limits=ignore_limits)
                    results_arr.append({
                        "method": method,
                        "repeat": repeat_idx,
                        "fold": fold_idx,
                        "sample": sample_idx,
                        "rmse": rmse,
                        "n_features": X_train.shape[1]
                    })                    
                    logger.info(
                        f"Repeat #{repeat_idx}, fold #{fold_idx}, sample #{sample_idx}, seed: {fold_random_state} "
                        f"---------------- RMSE:{rmse}"
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
    pd.DataFrame(results_arr).to_csv(temp_results_path, index=False)
    logger.info(f"Saved {method} results to {temp_results_path}")

    return results_arr


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Run TabPFN with optional sklearn feature selection or dimensionality reduction methods.")
    parser.add_argument("--dataset", type=str, default="363697", help="OpenML task ID or dataset name.")
    parser.add_argument("--method", type=str, nargs="+", default=["original"], choices=[
        "original", "random_fs", "variance_fs", "tree_fs", "pca_dr", "random_dr", "agglo_dr"
    ])
    parser.add_argument("--random_state", type=int, default=44)
    parser.add_argument("--max_num_feat", type=int, default=500)
    parser.add_argument("--var_threshold", type=float, default=0.95)
    parser.add_argument("--num_tree_estimators", type=int, default=15)
    parser.add_argument("--log_level", type=str, default="INFO", choices=[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ])

    args = parser.parse_args()

    # Set up config dict
    config = vars(args)

    # Set up logger
    logging.basicConfig(
    level=getattr(logging, config["log_level"].upper()), 
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__) 

    # Ignore TabPFN warnings about feature space dimension
    warnings.filterwarnings("ignore", message="Number of features .* is greater than the maximum.*")

    # Load OpenML dataset and task split
    try:
        #if args.dataset is the task ID
        task_id = int(args.dataset)
        task = openml.tasks.get_task(task_id)
    except ValueError: 
        #if args.dataset is the dataset name
        dataset_list = openml.datasets.list_datasets(output_format="dataframe")
        matching = dataset_list[dataset_list['name'] == args.dataset]
        
        if matching.empty:
            raise ValueError(f"No OpenML dataset found with name '{args.dataset}'")
        
        dataset_id = matching.iloc[0]['did']
        task_list = openml.tasks.list_tasks(output_format="dataframe", dataset=dataset_id)
        
        if task_list.empty:
            raise ValueError(f"No tasks found for dataset '{args.dataset}' (ID={dataset_id})")

        # Pick first matching task
        task_id = task_list.iloc[0]['tid']
        task = openml.tasks.get_task(task_id)

    dataset = task.get_dataset()
    config.update({
        "dataset_name": dataset.name
    })

    logger.info("")
    logger.info("Dataset:")
    logger.info("======================")
    logger.info(f"Loaded OpenML task: {task.task_id} - {task.task_type} on OpenML dataset {config["dataset_name"]}")

    # Retrieve splits and data from the task
    X, y = task.get_X_and_y(dataset_format="dataframe")
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    config.update({
        "n_repeats": n_repeats,
        "n_folds": n_folds,
        "n_samples": n_samples
    })
    logger.info(f"Dataset loaded with shape X: {X.shape}, y: {y.shape}")
    logger.info(f"Using split dimensions: repeats={n_repeats}, folds={n_folds}, samples={n_samples}")
    logger.info("======================")

    # Run TabPFN with selected method(s)
    all_results = []
    for method in config["method"]:
        results_method = run_training(X, y, method, config)
        all_results.extend(results_method)

    # Save results to csv
    os.makedirs("results", exist_ok=True)
    results_path = f"results/{config["dataset_name"]}_all.csv"

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved all results to {results_path}")

    # Save config to json
    config_path = f"results/{config['dataset_name']}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved config to {config_path}")