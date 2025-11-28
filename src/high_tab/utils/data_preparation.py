import openml
import re
import logging

import pandas as pd

logger = logging.getLogger(__name__) 

def prepare_data(config):
    if config["openml_id"]:
        task, dataset = load_dataset(config["openml_id"])
        task_type, task_type_log = get_task_type(task)
        eval_metric = get_eval_metric(task_type)

        config.update({
            "dataset_name": dataset.name,
            "task_type": task_type,
            "eval_metric": eval_metric,
        })

        logger.info("")
        logger.info("Dataset:")
        logger.info("======================")
        logger.info("OpenML task: %s", task.task_id)
        logger.info("Task type: %s", task_type_log)
        logger.info("Dataset name: %s", dataset.name)

        X, y = task.get_X_and_y(dataset_format="dataframe")
        config["ignore_limits"] = True if X.shape[1] >500 else False
        logger.info("Dataset size: %s, %s", X.shape, y.shape)

        if config["dry_run"]:
            config["num_repeats"], config["num_folds"] = (2, 2)
        else: # openml default or user-defined values
            num_repeats_default, num_folds_default, _ = task.get_split_dimensions()
            config["num_repeats"] = config.get("num_repeats") or num_repeats_default
            config["num_folds"]   = config.get("num_folds")   or num_folds_default
        
        return X, y, task, None
    
    if config["csv_path"]:
        target = config["target"] 
        df = pd.read_csv(config["csv_path"], low_memory=False, na_values=['NaN', "nan", "na", 'null', '', '?'])

        if config["test_csv_path"]:
            test_df = pd.read_csv(config["test_csv_path"], low_memory=False, na_values=['NaN', "nan", "na", 'null', '', '?'])
            test_df_idx = len(df) 
            df = pd.concat([df, test_df], ignore_index=True) # merge train and test sets
        else:
            test_df_idx = None

        y = df[target]
        X = df.drop(target,axis=1) 

        num_classes = len(y.unique())
        if num_classes==2:
            task_type = 'binary'
            task_type_log = "Binary classification"
        elif num_classes > 2 and num_classes < 20: #TODO: check for regression
            task_type = 'multiclass'
            task_type_log = f"Multiclass classification with {num_classes} classes"
        
        eval_metric = get_eval_metric(task_type)
        config["ignore_limits"] = True if X.shape[1] >500 else False
        config.update({
            "dataset_name": config["csv_path"],
            "task_type": task_type,
            "eval_metric": eval_metric,
        })

        if config["dry_run"]:
            config["num_repeats"], config["num_folds"] = (2, 2)
        else:
            config["num_repeats"], config["num_folds"] = (10, 3)

        logger.info("")
        logger.info("CSV dataset:")
        logger.info("======================")
        logger.info("Dataset name: %s", config["csv_path"])
        logger.info("Task type: %s", task_type_log)
        logger.info("Dataset size: %s, %s", X.shape, y.shape)

        return X, y, None, test_df_idx


def load_dataset(openml_id):
    """
    Args:
    - openml_id(str): Task ID or dataset name
    """
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
    task_type_str = task.task_type 
    task_type = None
    if re.search(r"regression", task_type_str.lower()):
        task_type = "regression"
        task_type_log = "Regression"
    elif re.search(r"classification", task_type_str.lower()):
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
        return ["rmse", "norm_rmse"]
    elif task_type == "binary":
        return ["roc_auc"]
    elif task_type == "multiclass":
        return ["log_loss"]
    else:
        raise ValueError(f"No evaluation metric found for task type: {task_type}")
