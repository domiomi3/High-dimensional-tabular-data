import argparse

import numpy as np
import pandas as pd

from scipy.io import arff
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
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

    print("RMSE:", np.sqrt(mse))
    print("----------------------------------------\n")


def prepare_dataset(df, method=None):
    X = df.drop(columns=TARGET)
    y = df[TARGET].astype(float)

    if method == "original":
        print("CATEGORY: Original")
        print("METHOD: None")
        print(f"Using original dataset with {X.shape[1]} features.\n")

    elif method == "random_fs":
        print("CATEGORY: Feature Selection")
        print(f"METHOD: Random selection of {MAX_NO_FEAT} features.\n")
        X = X.sample(n=MAX_NO_FEAT, axis=1, random_state=RANDOM_STATE)
        
    elif method == "variance_fs":
        print("CATEGORY: Feature Selection")
        print(f"METHOD: Variance threshold of {VARIANCE_THRESHOLD}.\n")
        var_FS = VarianceThreshold(threshold=(VARIANCE_THRESHOLD * (1 - VARIANCE_THRESHOLD)))
        X = var_FS.fit_transform(X)

    elif method == "tree_fs":
        print("CATEGORY: Feature Selection")
        print(f"METHOD: Feature importance computed from {TREE_ESTIMATORS} tree-based estimators.\n")
        clf = ExtraTreesClassifier(n_estimators=TREE_ESTIMATORS)
        clf = clf.fit(X, y.astype(int))
        model = SelectFromModel(clf, prefit=True)
        X = model.transform(X)

    elif method == "pca_dr":
        print("CATEGORY: Dimensionality Reduction")
        print(f"METHOD: PCA to {MAX_NO_FEAT} feature dimensions.\n")
        pca = PCA(n_components=MAX_NO_FEAT)
        X = pca.fit_transform(X)

    elif method == "random_dr":
        print("CATEGORY: Dimensionality Reduction")
        print(f"METHOD: Gaussian random projection to {MAX_NO_FEAT} feature dimensions.\n")
        transformer = GaussianRandomProjection(random_state=RANDOM_STATE, n_components=MAX_NO_FEAT)
        X = transformer.fit_transform(X)

    elif method == "agglo_dr":
        print("CATEGORY: Dimensionality Reduction")
        print(f"METHOD: Feature agglomeration to {MAX_NO_FEAT} clusters.\n")
        agglo = FeatureAgglomeration(n_clusters=MAX_NO_FEAT)
        X = agglo.fit_transform(X)

    else:
        print(f"{method} method unkown.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape, "\n")    

    return X_train, y_train, X_test, y_test

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Run TabPFN with optional sklearn feature selection or dimensionality reduction methods.")
    parser.add_argument("--dataset", type=str, required=True, default="dataset", help="Path to the ARFF dataset file.")
    parser.add_argument("--method", type=str, nargs="+", default=["original"], choices=[
        "original", "random_fs", "variance_fs", "tree_fs", "pca_dr", "random_dr", "agglo_dr"
    ])
    parser.add_argument("--target", type=str, default="MEDIAN_PXC50")
    parser.add_argument("--random_state", type=int, default=44)
    parser.add_argument("--max_no_feat", type=int, default=500)
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--variance_threshold", type=float, default=0.95)
    parser.add_argument("--tree_estimators", type=int, default=15)
    
    args = parser.parse_args()

    # Load QSAR-TID-11 dataset
    data, meta = arff.loadarff(args.dataset)
    df = pd.DataFrame(data)

    # Set global variables
    TARGET = args.target
    RANDOM_STATE = args.random_state
    MAX_NO_FEAT = args.max_no_feat
    TEST_SIZE = args.test_size
    VARIANCE_THRESHOLD = args.variance_threshold
    TREE_ESTIMATORS = args.tree_estimators

    # Run TabPFN with selected method(s)
    for method in args.method:
        X_train, y_train, X_test, y_test = prepare_dataset(df, method=method)
        print(f"NAME: {method}")
        if method == "original":
            run_tabpfn(X_train, y_train, X_test, y_test, ignore_pretraining_limits=True)
        else:
            run_tabpfn(X_train, y_train, X_test, y_test)

