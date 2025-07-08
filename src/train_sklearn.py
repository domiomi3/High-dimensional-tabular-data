#!/usr/bin/env python3

import argparse
import gc
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from preprocess_sklearn import *         # noqa: F401,F403
sys.path.append(".")
from utils import *                      # noqa: F401,F403


MODEL_DISPLAY = {
    "tabpfnv2_org": "TabPFNv2 original implementation",
    "tabpfnv2_tab": "TabPFNv2 TabArena implementation",
    "catboost_tab": "CatBoost TabArena implementation",
}

# ---------------------------------------------------------------------------#
#  Inference wrappers                                                        #
# ---------------------------------------------------------------------------#
def infer_tabarena_tabpfn(X_tr, y_tr, X_te, y_te,
                          task_type, eval_metric, eval_func):
    from external.tabrepo.tabrepo.models.utils import get_configs_generator_from_name

    model_meta = get_configs_generator_from_name(model_name="TabPFNv2")
    model_cls  = model_meta.model_cls
    eval_metric_ag = "root_mean_squared_error" if "rmse" in eval_metric \
                     else eval_metric[0]

    model = model_cls(problem_type=task_type,
                      eval_metric=eval_metric_ag,
                      name="TabPFNv2", path="models")
    model.fit(X=X_tr, y=y_tr, num_gpus=1)
    y_pred = model.predict(X_te) if task_type != "multiclass" \
            else model.predict_proba(X_te)
    model_cleanup(model)
    return eval_func(y_te, y_pred)


def infer_tabpfn(X_tr, y_tr, X_te, y_te,
                 task_type, ignore_pretraining_limits=False):
    from tabpfn import TabPFNRegressor, TabPFNClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if task_type == "regression":
        model = TabPFNRegressor(ignore_pretraining_limits=ignore_pretraining_limits,
                                device=device)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        model_cleanup(model)
        return root_mean_squared_error(y_te, preds)

    if task_type in {"binary", "multiclass"}:
        model = TabPFNClassifier(ignore_pretraining_limits=ignore_pretraining_limits,
                                 device=device)
        model.fit(X_tr, y_tr)
        if task_type == "multiclass":
            probs = model.predict_proba(X_te)
            model_cleanup(model)
            return log_loss(y_te, probs)
        preds = model.predict(X_te)
        model_cleanup(model)
        return roc_auc_score(y_te, preds)

    raise ValueError(f"Unknown task type: {task_type}")


def infer_catboost_tab(*_):
    raise NotImplementedError("CatBoost pipeline not yet integrated.")


# ---------------------------------------------------------------------------#
#  Core training loop                                                        #
# ---------------------------------------------------------------------------#
def train(X, y, task, method, cfg, logger) -> List[dict]:
    n_rep, n_fold, n_samp = cfg["n_repeats"], cfg["n_folds"], cfg["n_samples"]
    task_type   = cfg["task_type"]
    eval_metric = cfg["eval_metric"]
    eval_func   = cfg["eval_func"]
    model_key   = cfg["model"]
    check_time  = cfg["check_time"]

    results: List[dict] = []
    fold_times: List[float] = []
    first_banner = True

    logger.info("")             # blank line
    logger.info("Preprocessing:")
    logger.info("======================")

    for rep in range(n_rep):
        rep_seed = cfg["random_state"] + 100 * rep
        for fold in range(n_fold):
            for samp in range(n_samp):
                if check_time:
                    t0 = time.time()

                try:
                    tr_idx, te_idx = task.get_train_test_split_indices(
                        repeat=rep, fold=fold, sample=samp)
                    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

                    X_tr, X_te, y_tr, y_te = run_preprocessing_pipeline(
                        X_tr, X_te, y_tr, y_te, method, task_type,
                        rep_seed, cfg, logger, first_banner)

                    if first_banner:
                        logger.info(
                            "Dataset size: X_train %s  y_train %s  "
                            "X_test %s  y_test %s",
                            X_tr.shape, y_tr.shape, X_te.shape, y_te.shape
                        )
                        if X_tr.shape[1] > 500:
                            logger.warning("Number of features >500 "
                                           "→ TabPFN may underperform.")
                        logger.info("") 
                        logger.info("Training")
                        logger.info("======================")
                        logger.info("Model: %s", MODEL_DISPLAY.get(model_key, model_key))
                        logger.info("") 
                        first_banner = False

                    ignore_limits = X_tr.shape[1] > 500

                    if model_key == "tabpfnv2_tab":
                        score = infer_tabarena_tabpfn(
                            X_tr, y_tr, X_te, y_te,
                            task_type, eval_metric, eval_func)
                    elif model_key == "tabpfnv2_org":
                        score = infer_tabpfn(
                            X_tr, y_tr, X_te, y_te,
                            task_type, ignore_pretraining_limits=ignore_limits)
                    elif model_key == "catboost_tab":
                        score = infer_catboost_tab()
                    else:
                        raise ValueError(f"Unknown model: {model_key}")

                    extra = {}
                    if task_type == "regression" and "norm_rmse" in eval_metric:
                        extra["norm_rmse"] = score / np.std(y_te)

                    results.append({
                        "dataset_name": cfg["dataset_name"],
                        "method": method,
                        "repeat": rep,
                        "fold": fold,
                        "sample": samp,
                        "n_features": X_tr.shape[1],
                        eval_metric[0]: score,
                        **extra,
                    })

                    logger.info(
                        "repeat %d | fold %d | sample %d | %s: %.4f%s",
                        rep, fold, samp, eval_metric[0], score,
                        f" | normRMSE: {extra['norm_rmse']:.4f}" if extra else ""
                    )

                    del X_tr, X_te, y_tr, y_te
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error("Failure r%02d f%02d s%02d m%-10s",
                                 rep, fold, samp, method)
                    logger.exception(e)

                finally:
                    if check_time:
                        fold_times.append(time.time() - t0)

    avg_fold = float(np.mean(fold_times)) if fold_times else None
    total_sec = float(sum(fold_times)) if fold_times else None
    if check_time:
        logger.info("Average fold time for '%s': %.1fs over %d folds",
                    method, avg_fold, len(fold_times))

    for row in results:
        row["avg_fold_time"]      = avg_fold
        row["total_elapsed_time"] = total_sec

    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    save_results_to_csv(results, csv_path, logger)

    yaml_path = out_dir / "config.yaml"
    save_yaml({
        **cfg,
        "method": method,
        "avg_fold_time": avg_fold,
        "total_elapsed_time": total_sec,
    }, yaml_path, logger)

    return results


# ---------------------------------------------------------------------------#
#  Dataset-level orchestration                                               #
# ---------------------------------------------------------------------------#
def run_on_dataset(cfg, logger):
    task, dataset = load_dataset(cfg["openml_id"])
    task_type, task_type_log = get_task_type(task)
    eval_metric, eval_func   = get_eval_metric(task_type)

    cfg.update({
        "dataset_name": dataset.name,
        "task_type": task_type,
        "eval_metric": eval_metric,
        "eval_func": eval_func,
    })

    logger.info("")
    logger.info("Dataset:")
    logger.info("======================")
    logger.info("OpenML task : %s", task.task_id)
    logger.info("Task type   : %s", task_type_log)
    logger.info("Dataset name: %s", dataset.name)

    X, y = task.get_X_and_y(dataset_format="dataframe")
    logger.info("Data shape  : %s, %s", X.shape, y.shape)

    if cfg["dry_run"]:
        cfg["n_repeats"], cfg["n_folds"], cfg["n_samples"] = (2, 2, 1)
    else:
        cfg["n_repeats"], cfg["n_folds"], cfg["n_samples"] = \
            task.get_split_dimensions()
    logger.info("Splits      : repeats=%d folds=%d samples=%d",
                cfg["n_repeats"], cfg["n_folds"], cfg["n_samples"])

    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    for m in cfg["method"]:
        all_rows.extend(train(X, y, task, m, cfg, logger))

    if len(cfg["method"]) > 1:
        out_dir = Path(cfg["results_dir"])
        save_yaml({**cfg, "methods": cfg["method"]},
                  out_dir / "config.yaml", logger)
        save_results_to_csv(all_rows,
                            out_dir / f"results.csv", logger)

    df = pd.DataFrame(all_rows)
    summary = df.groupby("method")[cfg["eval_metric"]].agg(["mean", "std"]).round(4)
    logger.info("\nSummary:\n%s", summary.to_string())


# ---------------------------------------------------------------------------#
#  Main                                                                       #
# ---------------------------------------------------------------------------#
def main(cfg):
    logger = setup_logger(cfg["log_level"].upper())
    ensure_gpu_or_die(logger)

    warnings.filterwarnings("ignore",
        message="Number of features .* is greater than the maximum.*")
    warnings.filterwarnings("ignore",
        message="pkg_resources is deprecated as an API.*", category=UserWarning)
    warnings.filterwarnings("ignore",
        message="X does not have valid feature names.*", category=UserWarning)

    t0 = time.time()
    for oid in cfg["openml_id"]:
        run_cfg = cfg.copy()
        run_cfg["openml_id"] = oid
        run_on_dataset(run_cfg, logger)

    hrs, rem = divmod(time.time() - t0, 3600)
    mins, secs = divmod(rem, 60)
    logger.info("Total elapsed: %dh %dm %.1fs", int(hrs), int(mins), secs)


# ---------------------------------------------------------------------------#
#  Entry-point                                                                #
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    ALL_METHODS = [
        "original", "random_fs", "variance_fs", "tree_fs", "kbest_fs",
        "pca_dr", "random_dr", "agglo_dr", "ica_dr", "kpca_dr",
    ]

    ap = argparse.ArgumentParser(description="Run TabPFNv2 / CatBoost + FS/DR pipeline")
    ap.add_argument("--openml_id", nargs="+", default=["363697"])
    ap.add_argument("--method", nargs="+", default=["original", "tree_fs"],
                    help="'all' to run every method.")
    ap.add_argument("--model", default="tabpfnv2_org",
                    choices=["tabpfnv2_org", "tabpfnv2_tab", "catboost_tab"])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--exp_name", default="exp")      # kept though not used in path
    ap.add_argument("--metric_override", type=str)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--check_time", action="store_true")
    ap.add_argument("--random_state", type=int, default=44)
    ap.add_argument("--n_features", type=int, default=500)
    ap.add_argument("--var_threshold", type=float, default=0.95)
    ap.add_argument("--n_tree_estimators", type=int, default=15)
    ap.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = ap.parse_args()
    cfg = vars(args)
    cfg["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "all" in cfg["method"]:
        cfg["method"] = ALL_METHODS

    main(cfg)
