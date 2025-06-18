import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression, SelectFromModel
)
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration


def encode_categorical_fit(X_train):
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_enc = X_train.copy()
    cat_cols = X_train_enc.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        X_train_enc[cat_cols] = encoder.fit_transform(X_train_enc[cat_cols]).astype(int)
    return pd.DataFrame(X_train_enc), encoder, cat_cols


def encode_categorical_transform(X_test, encoder, cat_cols, metadata_flag, logger):
    X_test_enc = X_test.copy()
    if len(cat_cols) > 0:
        X_test_enc[cat_cols] = encoder.transform(X_test_enc[cat_cols]).astype(int)
    if metadata_flag:
        logger.info(f"Encoded {len(cat_cols)} categorical features.")
    return pd.DataFrame(X_test_enc)


def remove_const_columns(X_train, X_test, metadata_flag, logger):
    cols = X_train.shape[1]
    vt = VarianceThreshold(threshold=0.0)  # Remove features with 0 variance
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)
    if metadata_flag:
        logger.info(f"Removed {cols-X_train.shape[1]} features with 0 variance.")

    return pd.DataFrame(X_train), pd.DataFrame(X_test)


def normalize_numeric(X_train, X_test, num_cols, metadata_flag, logger):
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    if metadata_flag:
        logger.info(f"Standardized {len(num_cols)} numeric features.")
    return pd.DataFrame(X_train), pd.DataFrame(X_test)


def encode_cateogrical_labels(y_train, y_test, metadata_flag, logger):
    # Convert categorical string labels ("yes"/"no") to binary integers
    if y_train.dtype == "object" or y_train.dtype.name == "category":
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train).ravel(), index=y_train.index)
        y_test = pd.Series(le.transform(y_test).ravel(), index=y_test.index)
        if metadata_flag:
            logger.info("Categorical labels converted to numerical format.")
    return y_train, y_test


def apply_sklearn_method(X_train, X_test, y_train, metadata_flag, method, random_state, config, logger):
    n_features = config["n_features"]
    n_tree_estimators = config["n_tree_estimators"]
    var_threshold = config["var_threshold"]
    task_type = config["task_type"]

    if method == "original":
        if metadata_flag:
            logger.info("Category: Original")
            logger.info("Method: None")
            logger.info(f"Using original dataset with {X_train.shape[1]} features.")

    elif method == "random_fs":
        if metadata_flag:
            logger.info("Category: Feature Selection")
            logger.info(f"Method: Random selection")
            logger.info(f"HPs: Maximum number of features: {n_features}")
        columns = X_train.sample(n=n_features, axis=1, random_state=random_state).columns
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
            logger.info(f"HPs: Number of estimators: {n_tree_estimators}")
            # breakpoint()
        if task_type == "regression":
            model = ExtraTreesRegressor(n_estimators=n_tree_estimators)
        elif task_type == "classification":
            model = ExtraTreesClassifier(n_estimators=n_tree_estimators)
        model.fit(X_train, y_train)
        model = SelectFromModel(model, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)

    elif method == "kbest_fs": # requires non-constant columns
        if metadata_flag:
            logger.info("Category: Feature Selection")
            logger.info(f"Method: Select K Best")
            logger.info(f"HPs: Number of top features: {n_features}")

        stat_test = f_regression if task_type == "regression" else f_classif
        kbest = SelectKBest(stat_test, k=n_features)
        X_train = kbest.fit_transform(X_train, y_train)
        X_test = kbest.transform(X_test)

    elif method == "pca_dr":
        if metadata_flag:
            logger.info("Category: Dimensionality Reduction")
            logger.info(f"Method: Principal Component Analysis")
            logger.info(f"HPs: Maximum number of features: {n_features}")
        pca = PCA(n_components=n_features, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        # thresholds = [0.50, 0.75, 0.90, 0.95]
        # cumulative_variance = pca.explained_variance_ratio_.cumsum()
        # threshold_components = {
        #     f"components_for_{int(t * 100)}": int(np.argmax(cumulative_variance >= t) + 1)
        #     for t in thresholds
        # }

        # # Save threshold info to CSV
        # save_path = os.path.join(config.get("output_dir", "."), f"pca_thresholds.csv")
        # threshold_components["dataset"] = config.get("dataset_id", "unknown")
        # threshold_components["original_num_features"] = X_train.shape[1]

        # # Append or create file
        # df_new = pd.DataFrame([threshold_components])
        # if os.path.exists(save_path):
        #     df_existing = pd.read_csv(save_path)
        #     df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        #     df_combined.to_csv(save_path, index=False)
        # else:
        #     df_new.to_csv(save_path, index=False)

    elif method == "random_dr":
        if metadata_flag:
            logger.info("Category: Dimensionality Reduction")
            logger.info(f"Method: Gaussian random projection")
            logger.info(f"HPs: Maximum number of features: {n_features}")
        transformer = GaussianRandomProjection(n_components=n_features, random_state=random_state)
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)

    elif method == "agglo_dr":
        if metadata_flag:
            logger.info("Category: Dimensionality Reduction")
            logger.info(f"Method: Feature agglomeration")
            logger.info(f"HPs: Maximum number of clusters: {n_features}")
        agglo = FeatureAgglomeration(n_clusters=n_features)
        X_train = agglo.fit_transform(X_train)
        X_test = agglo.transform(X_test)

    elif method == "ica_dr":
        if metadata_flag:
            logger.info("Category: Dimensionality Reduction")
            logger.info(f"Method: Independent Component Analysis")
            logger.info(f"HPs: Number of components: {n_features}")
        ica = FastICA(n_components=n_features, random_state=random_state)
        X_train = ica.fit_transform(X_train)
        X_test = ica.transform(X_test)

    elif method == "kpca_dr":
        if metadata_flag:
            logger.info("Category: Dimensionality Reduction")
            logger.info(f"Method: Kernel PCA")
            logger.info(f"HPs: Number of components: {n_features}, Kernel: RBF")
        kpca = KernelPCA(n_components=n_features, kernel="rbf", random_state=random_state)
        X_train = kpca.fit_transform(X_train)
        X_test = kpca.transform(X_test)

    else:
        logger.warning(f"{method} method unknown.")

    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.Series(y_train)


def run_preprocessing_pipeline(
    X_train, X_test, y_train, y_test, method, task_type,
    random_state, config, logger, metadata_flag
):
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    # 1. Remove constant features
    X_train, X_test = remove_const_columns(X_train, X_test, metadata_flag, logger)

    # 2. Label binarization (only for classification)
    #TODO: LabelCleaner from AutoGluon
    #TODO: prob should be done elsewhere
    if task_type == "classification":
        y_train, y_test = encode_cateogrical_labels(y_train, y_test, metadata_flag, logger)

    # 3. Encode categoricals (for selected methods only)
    if method in ["variance_fs", "tree_fs", "kbest_fs", "pca_dr", "random_dr", "agglo_dr", "ica_dr", "kpca_dr"]:
        X_train, encoder, cat_cols = encode_categorical_fit(X_train)
        X_test = encode_categorical_transform(X_test, encoder, cat_cols, metadata_flag, logger)

    # 4. Normalize numeric columns for DR methods (after encoding)
        if method in ["pca_dr", "ica_dr", "kpca_dr", "agglo_dr"]:
            num_cols = [col for col in X_train.columns if col not in cat_cols] 
            if num_cols: # otherwise normalizing encoded categorical features
                X_train, X_test = normalize_numeric(X_train, X_test, num_cols, metadata_flag, logger)

    # 5. Apply selected feature selection or dimensionality reduction
    X_train, X_test, y_train = apply_sklearn_method(
        X_train, X_test, y_train, metadata_flag, method, random_state, config, logger
    )

    return X_train, X_test, y_train, y_test
