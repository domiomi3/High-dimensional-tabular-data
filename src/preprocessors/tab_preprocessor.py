import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression, SelectFromModel
)
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration

from .sand_layer import SANDProcessor

class TabPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, rand_state: int = 0, **config):
        self.rand_state = rand_state
        self.config = config
        self._estimator = None # sklearn method

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        y = None if y is None else pd.Series(y)       

        config = self.config   
        method = config["method"]     

        if method == "original":
            self._estimator = None              
        elif method == "random_fs":
            self._estimator = list(
                X.sample(n=config["num_features"], axis=1, random_state=self.rand_state).columns
            )
        elif method == "variance_fs":
            self._estimator = VarianceThreshold(
                threshold=config["var_threshold"] * (1 - config["var_threshold"])
            ).fit(X, y)
        elif method == "tree_fs":
            base = (
                ExtraTreesRegressor if config["task_type"] == "regression"
                else ExtraTreesClassifier
            )(n_estimators=config["num_tree_estimators"], random_state=self.rand_state)
            base.fit(X, y)
            self._estimator = SelectFromModel(base, prefit=True)
        elif method == "kbest_fs":
            stat = f_regression if config["task_type"] == "regression" else f_classif
            self._estimator = SelectKBest(stat, k=config["num_features"]).fit(X, y)
        elif method == "pca_dr":
            self._estimator = PCA(n_components=config["num_features"],
                                   random_state=self.rand_state).fit(X)
        elif method == "random_dr":
            self._estimator = GaussianRandomProjection(n_components=config["num_features"], random_state=self.rand_state).fit(X)
        elif method == "agglo_dr":
            self._estimator = FeatureAgglomeration(n_clusters=config["num_features"]).fit(X)
        elif method == "kpca_dr":
            self._estimator = KernelPCA(n_components=config["num_features"], kernel="rbf", random_state=self.rand_state).fit(X)
        elif method == "kbest+pca":
            # 1 k best
            stat = f_regression if config["task_type"] == "regression" else f_classif

            self._selector = SelectKBest(stat, k=config["num_kbest_features"]).fit(X, y)
            self._kbest_mask = self._selector.get_support() # selected columns
            # 2 PCA 
            remaining_cols = X.columns[~self._kbest_mask]
            self._pca_cols = remaining_cols                     

            if len(remaining_cols) == 0: # if all cols are selected
                self._pca = None
            else:
                self._pca = PCA(
                    n_components=config["num_pca_comps"],
                    random_state=self.rand_state,
                ).fit(X[remaining_cols])
        elif method == "sand_fs":
            self._estimator = SANDProcessor(
                num_features=config["num_features"], 
                device=config["device"],
                task_type=config["task_type"],
                random_state=self.rand_state
            ).fit(X, y)
        else:
            raise ValueError(f"Unknown method {method}")
            

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        method = self.config["method"]

        if method == "original":
            return X
        elif method == "random_fs":
            return X[self._estimator] # _estimator is just a list of cols
        elif method == "kbest+pca":
            kbest_arr = self._selector.transform(X)
            kbest_cols = X.columns[self._kbest_mask]
            kbest_df = pd.DataFrame(kbest_arr, columns=kbest_cols, index=X.index)

            if self._pca is not None:
                pca_arr = self._pca.transform(X[self._pca_cols])
                pca_cols = [f"PCA_{i}" for i in range(pca_arr.shape[1])]
                pca_df = pd.DataFrame(pca_arr, columns=pca_cols, index=X.index)
                return pd.concat([kbest_df, pca_df], axis=1)
            else:
                return kbest_df # if all cols selected
        elif method == "sand_fs":
            return X[self._estimator.selected_cols_]
        else:
            return pd.DataFrame(self._estimator.transform(X))

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)
