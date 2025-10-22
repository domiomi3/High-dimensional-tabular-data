import logging

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
    "sand_fs":      "SAND layer"
}

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
    else:
        logger.warning("Unknown preprocessing method %s", method)


def run_preprocessing_pipeline(
    X_train, X_test, y_train, y_test, random_state, config, metadata_flag
):
    preprocessing = config["preprocessing"] # model-agnostic or model-specific
    
    # AG tab data preprocessing
    feature_generator, label_cleaner = (
        AutoMLPipelineFeatureGenerator(),
        LabelCleaner.construct(problem_type=config["task_type"], y=y_train),

    )
    X_train, y_train = (
        feature_generator.fit_transform(X_train),
        label_cleaner.transform(y_train),
    )
    X_test, y_test = feature_generator.transform(X_test), label_cleaner.transform(y_test)

    # feature dimensionality reduction/selection
    if preprocessing == "model-agnostic":
        tab_preprocessor = TabPreprocessor(rand_state=random_state, **config) 
        X_train = tab_preprocessor.fit_transform(X=X_train, y=y_train)
        X_test = tab_preprocessor.transform(X_test)

    # log if first iteration
    if metadata_flag:
        logger.info("Using AutoGluon's AutoMLPipelineFeatureGenerator() and LabelCleaner().")
        if preprocessing == "model-agnostic":
            logger.info("Model-agnostic FS/DR.")
        else:
            logger.info("Model-specific (per-fold) FS/DR.")
        log_metadata(config)


    return X_train, X_test, y_train, y_test
