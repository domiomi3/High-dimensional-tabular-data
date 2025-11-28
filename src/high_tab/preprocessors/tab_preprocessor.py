import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression, SelectFromModel
)
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler


from high_tab.preprocessors.sand_layer import SANDProcessor
from high_tab.preprocessors.tabpfn_selector import TabPFNSelector

class TabPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, rand_state: int = 0, **config):
        self.rand_state = rand_state
        self.config = config
        self._estimator = None # sklearn method
        self._feature_names_out = None  # output feature names during fit

    def fit(self, X: pd.DataFrame, y=None):
        y = None if y is None else pd.Series(y)       

        config = self.config   
        method = config["method"]     

        if method == "original":
            self._estimator = None
            self._feature_names_out = X.columns.tolist()
            
        elif method == "random_fs":
            selected_cols = list(
                X.sample(n=config["num_features"], axis=1, random_state=self.rand_state).columns
            )
            self._estimator = selected_cols
            self._feature_names_out = selected_cols
            
        elif method == "variance_fs":
            self._estimator = VarianceThreshold(
                threshold=config["var_threshold"] * (1 - config["var_threshold"])
            ).fit(X, y)
            mask = self._estimator.get_support()
            self._feature_names_out = X.columns[mask].tolist()
            
        elif method == "tree_fs":
            base = (
                ExtraTreesRegressor if config["task_type"] == "regression"
                else ExtraTreesClassifier
            )(n_estimators=config["num_tree_estimators"], random_state=self.rand_state)
            base.fit(X, y)
            self._estimator = SelectFromModel(base, prefit=True)
            mask = self._estimator.get_support()
            self._feature_names_out = X.columns[mask].tolist()
            
        elif method == "kbest_fs":
            stat = f_regression if config["task_type"] == "regression" else f_classif
            self._estimator = SelectKBest(stat, k=config["num_features"]).fit(X, y)
            mask = self._estimator.get_support()
            self._feature_names_out = X.columns[mask].tolist()
            
        elif method == "pca_dr":
            self._estimator = PCA(n_components=config["num_features"],
                                   random_state=self.rand_state).fit(X)
            self._feature_names_out = [f"PCA_{i}" for i in range(config["num_features"])]
            
        elif method == "random_dr":
            self._estimator = GaussianRandomProjection(
                n_components=config["num_features"], 
                random_state=self.rand_state
            ).fit(X)
            self._feature_names_out = [f"RP_{i}" for i in range(config["num_features"])]
            
        elif method == "agglo_dr":
            self._estimator = FeatureAgglomeration(n_clusters=config["num_features"]).fit(X)
            self._feature_names_out = [f"Agglo_{i}" for i in range(config["num_features"])]
            
        elif method == "kpca_dr":
            self._estimator = KernelPCA(
                n_components=config["num_features"], 
                kernel="rbf", 
                random_state=self.rand_state
            ).fit(X)
            self._feature_names_out = [f"KPCA_{i}" for i in range(config["num_features"])]
            
        elif method == "kbest+pca":
            # 1. K-best selection
            stat = f_regression if config["task_type"] == "regression" else f_classif
            self._selector = SelectKBest(stat, k=config["num_kbest_features"]).fit(X, y)
            self._kbest_mask = self._selector.get_support()
            
            # 2. PCA on remaining features
            remaining_cols = X.columns[~self._kbest_mask]
            self._pca_cols = remaining_cols
            
            kbest_cols = X.columns[self._kbest_mask].tolist()
            
            if len(remaining_cols) == 0:
                self._pca = None
                self._feature_names_out = kbest_cols
            else:
                self._pca = PCA(
                    n_components=config["num_pca_comps"],
                    random_state=self.rand_state,
                ).fit(X[remaining_cols])
                pca_cols = [f"PCA_{i}" for i in range(config["num_pca_comps"])]
                self._feature_names_out = kbest_cols + pca_cols
                
        elif method == "sand_fs":
            self._estimator = SANDProcessor(
                num_features=config["num_features"], 
                device=config["device"],
                task_type=config["task_type"],
                random_state=self.rand_state,
                model_type=config["model_type"]
            ).fit(X, y)
            self._feature_names_out = self._estimator.selected_cols_
        elif method == "lasso_fs":
            X_scaled = StandardScaler(with_mean=True).fit_transform(X) #only for fitting
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index) 
            tol = 1e-3       
            max_iter = 3000
            
            if config["task_type"] == "regression":
                # heuristics: alpha_max = (1/n) * max |X^T y| on standardized X
                n = X_scaled.shape[0]
                y_center = y - np.mean(y) if y is not None else y
                alpha_max = (np.abs(X_scaled.T @ y_center).max() / n) if y is not None else 1.0
                alpha = config.get("alpha", 0.1 * alpha_max)  # 10% of alpha_max is a good starting point
                base = Lasso(alpha=alpha, tol=tol, max_iter=max_iter, warm_start=True)
            else:  # classification
                C = 1.0  # smaller C => stronger sparsity
                base = LogisticRegression(
                    penalty="l1", solver="saga", C=C, tol=tol, max_iter=max_iter,
                    warm_start=True
                )
            base.fit(X_scaled, y)

            self._estimator = SelectFromModel(
                estimator=base,
                threshold=0,  # keep any strictly non-zero coef
                max_features=config["num_features"],
                prefit=True
            )
            mask = self._estimator.get_support()
            self._feature_names_out = X.columns[mask].tolist()
        elif method == "tabpfn_fs":
            self._estimator = TabPFNSelector(
                num_features=config["num_features"], 
                device=config["device"],
            ).fit(X, y)
            self._feature_names_out = self._estimator.selected_cols_
        else:
            raise ValueError(f"Unknown method {method}")
            
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        method = self.config["method"]

        if method == "original":
            return X
            
        elif method == "random_fs":
            return X[self._estimator]
            
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
                return kbest_df
                
        elif method in ["tabpfn_fs","sand_fs"]:
            return X[self._estimator.selected_cols_]
        
        else:
            X_transformed = self._estimator.transform(X)
            return pd.DataFrame(
                X_transformed, 
                columns=self._feature_names_out, 
                index=X.index
            )

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)