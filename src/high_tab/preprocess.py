import logging
import pandas as pd
import torch
from typing import Any, Dict

logger = logging.getLogger(__name__)

for name in ("autogluon", "autogluon.core", "autogluon.features"):
    logging.getLogger(name).setLevel(logging.WARNING) 

from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.core.data import LabelCleaner

from high_tab.preprocessors.tab_preprocessor import TabPreprocessor

_METHOD_FULLNAME = {
    "random_fs":    "Random selection",
    "variance_fs":  "Variance threshold",
    "tree_fs":      "Tree-based estimators",
    "kbest_fs":     "K best selection",
    "pca_dr":       "PCA",
    "random_dr":    "Gaussian random projection",
    "agglo_dr":     "Feature agglomeration",
    "kpca_dr":      "Kernel PCA",
    "sand_fs":      "SAND layer",
    "lasso_fs":     "Lasso",
    "tabpfn_fs":    "TabPFN-wide+decoder"
}


def _normalize_type(df: pd.DataFrame, missing="NaN"):
    df = df.copy()
    cat_dtype_cols = df.select_dtypes(include=["category"]).columns
    if len(cat_dtype_cols):
        # catboost creates cat_features based on object type not pd's categroy
        df[cat_dtype_cols] = df[cat_dtype_cols].astype(object) 
    # obj_cols = df.select_dtypes(include=["object"]).columns
    # for c in obj_cols:
    #     if df[c].isna().any():
    #         df[c] = df[c].fillna(missing) #convert NaN to str
    #     df[c] = df[c].astype(str)
    return df


def log_metadata(config: Dict[str, Any]) -> None:
    method = config["method"]
    display_method = _METHOD_FULLNAME.get(method, method)

    if method == "original":
        logger.info("Category: Original")
        logger.info("Method: None")
        logger.info("Using original dataset.")
        return

    if method == "kbest+pca":
        num_kbest_features = config["num_kbest_features"]
        num_pca_comps = config["num_pca_comps"]
        logger.info("Category: Feature selection + Dimensionality reduction")
        logger.info("Method: K best selection + PCA")
        logger.info(f"HPs: Top-k features(75%): {num_kbest_features}, PCA components(25%): {num_pca_comps}")
        return
    
    cat = ("Feature selection" if method.endswith("_fs")
           else "Dimensionality reduction")
    logger.info("Category: %s", cat)
    logger.info("Method: %s", display_method)

    num_feat = config["num_features"]
    vt = config["var_threshold"]
    num_trees= config["num_tree_estimators"]
    num_ensemble = config["num_ensemble"]

    if method == "random_fs":
        logger.info("HPs: Maximum number of features: %d", num_feat)
    elif method == "variance_fs":
        logger.info("HPs: Threshold: %.3f", vt)
    elif method == "tree_fs":
        logger.info("HPs: Number of estimators: %d", num_trees)
    elif method == "kbest_fs":
        logger.info("HPs: Top-k features: %d", num_feat)
    elif method in {"pca_dr", "random_dr", "agglo_dr", "kpca_dr"}:
        logger.info("HPs: Components/features: %d", num_feat)
        if method == "kpca_dr":
            logger.info("Kernel: RBF")
    elif method == "sand_fs":
        logger.info("HPs: Number of features: %d", num_feat)
    elif method == "lasso_fs":
            logger.info("HPs: Number of features: %d", num_feat)
    elif method == "tabpfn_fs":
            logger.info("HPs: Number of features: %d", num_feat)
            logger.info("HPs: Number of estimators: %d", num_ensemble)
    else:
        logger.warning("Unknown preprocessing method %s", method)


def impute_nans(X, metadata_flag=False, is_train=True):
    """
    Impute NaNs using statistics:
    - Numeric: mean (0.0 if all NaN)
    - Categorical/Object: mode ("missing" if all NaN)
    
    """
    X_imp = X.copy()
    num_nans = X.isna().sum().sum()

    # numeric -> mean
    num_cols = X.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        means = X[num_cols].mean().fillna(0.0) # numeric stability whe all nans
        X_imp[num_cols] = X_imp[num_cols].fillna(means)
    
    # cat -> mode
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            mode = X[col].mode()[0] if len(X[col].mode()) > 0 else 'missing'
            X_imp[col] = X_imp[col].fillna(mode)

    if metadata_flag:
        if num_nans > 0:
            set_type = "training" if is_train else "test"
            logger.info(f"Imputed {num_nans} missing values for {set_type} dataset.")
    
    return X_imp

def run_preprocessing_pipeline(
    X_train, X_test, y_train, y_test, random_state, config, metadata_flag
):
    preprocessing = config["preprocessing"] # model-agnostic or model-specific

    X_train = impute_nans(X_train, metadata_flag)
    X_test = impute_nans(X_test, metadata_flag, is_train=False)

    # AG tab data preprocessing
    feature_generator, label_cleaner = (
        AutoMLPipelineFeatureGenerator(),
        LabelCleaner.construct(problem_type=config["task_type"], y=y_train),
    )
    # low cardinality -> int8, high -> cat
    X_train, y_train = (
        feature_generator.fit_transform(X_train),
        label_cleaner.transform(y_train),
    )
    if metadata_flag: # log if first iteration
        logger.info("Using AutoGluon's AutoMLPipelineFeatureGenerator() and LabelCleaner().")

    X_test, y_test = feature_generator.transform(X_test), label_cleaner.transform(y_test)
    
    # AG-produces nans
    X_train = impute_nans(X_train, metadata_flag)
    X_test = impute_nans(X_test, metadata_flag, is_train=False)

    # change category type to object
    if config["model"] == "catboost_tab":
        X_train = _normalize_type(X_train)
        X_test  = _normalize_type(X_test)

    # feature dimensionality reduction/selection
    if preprocessing == "model-agnostic":
        tab_preprocessor = TabPreprocessor(rand_state=random_state, **config) 
        X_train = tab_preprocessor.fit_transform(X=X_train, y=y_train)
        X_test = tab_preprocessor.transform(X_test)

    if metadata_flag:
        if preprocessing == "model-agnostic":
            logger.info("Model-agnostic FS/DR.")
        else:
            logger.info("Model-specific (per-fold) FS/DR.")
        log_metadata(config)

    return X_train, X_test, y_train, y_test
