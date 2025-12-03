import torch
import logging
import pandas as pd

from lasso_prior.model.decoder import TabPFNFeatureSelector
from high_tab.utils.hardware import log_mem, memory_cleanup

logger = logging.getLogger(__name__)

_TABPFN_MODEL_CACHE = {}

import numpy as np
import torch

from tabpfn import TabPFNClassifier
from torch.utils.data import DataLoader

class TabPFNPreprocessor:
    """
    Preserves feature dimensionality and provides n_ensemble different preprocessed X.
    """
    def __init__(
        self,
        n_ensemble: int = 8,
        device: str | None = None,
        random_state: int = 44,
        max_data_size: int | None = None,
    ):
        self.n_ensemble = n_ensemble
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.max_data_size = max_data_size
        self._clf = None  # lazy init

    def _setup_tabpfn_model(self):
        if self._clf is not None:
            return self._clf
        
        inference_config = {
            "PREPROCESS_TRANSFORMS": [
                {
                    "name": "none", 
                    "append_original": False,       
                    "categorical_name": "numeric",  
                    "global_transformer_name": None,
                    "max_features_per_estimator": 10000 # otherwise feature dim always reduced to 500!
                },
            ],
            "FINGERPRINT_FEATURE": False,
        }

        base = TabPFNClassifier()
        clf = base.create_default_for_version(
            version="v2",
            device=self.device,
            n_estimators=self.n_ensemble,
            ignore_pretraining_limits=True,
            inference_precision=torch.float16,
            inference_config=inference_config,
            random_state=self.random_state,
        )

        clf._initialize_model_variables()

        for model in clf.models_:
            model.features_per_group = 1

        self._clf = clf
        return self._clf

    @staticmethod
    def _to_numpy(X, y):
        # X can be df or array, y can be series or array
        if hasattr(X, "to_numpy"):
            X_np = X.to_numpy(dtype=np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        if hasattr(y, "to_numpy"):
            y_np = y.to_numpy()
        else:
            y_np = np.asarray(y)

        _, y_np = np.unique(y_np, return_inverse=True)
        return X_np.astype(np.float32), y_np.astype(np.int64)

    @staticmethod
    def _fake_split(X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        X_test = X[:1]  # (1, n_features)
        y_test = y[:1]
        return X, X_test, y, y_test

    def generate_preprocessed_datasets(self, X, y):
        """
        """
        clf = self._setup_tabpfn_model()
        X_np, y_np = self._to_numpy(X, y)

        datasets_collection = clf.get_preprocessed_datasets(
            X_np,
            y_np,
            self._fake_split,
            max_data_size=self.max_data_size,
        )

        from tabpfn.utils import meta_dataset_collator
        loader = DataLoader(
            datasets_collection,
            batch_size=1,
            collate_fn=meta_dataset_collator,
        )
        (
            X_trains_preprocessed,   
            X_tests_preprocessed,    
            y_trains_preprocessed,   
            y_tests_preprocessed,    
            cat_ixs,
            confs,
        ) = next(iter(loader))

        # We only need X_trains_preprocessed: list[Tensor], each (n_samples, n_features)
        return X_trains_preprocessed, y_trains_preprocessed


class TabPFNSelector:
    """
    For preprocessing, use selected_columns_ (similar to random_fs method).
    """
    def __init__(self, num_features, num_ensemble, emb_layer, random_state, device):
        self.K = int(num_features)
        self.checkpoint_dir = "decoder_models/train_lasso_4000_20251120_231955/"
        self.model_name = "best_model"
        self.selected_cols_ = None
        self.device = "cpu"
        self.emb_layer = emb_layer
        self.model = self._load_model()
        self.num_ensemble = num_ensemble
        self.random_state = random_state
        self.preprocessor = TabPFNPreprocessor(
            n_ensemble=self.num_ensemble,
            device=self.device,
            random_state=self.random_state,
        )

    def _load_model(self):
        model = TabPFNFeatureSelector(
            model_name="TabPFN-Wide-5k",
            model_checkpoint_dir="external/tabpfnwide/models",
            embedding_layer=self.emb_layer, 
            device=self.device,
        )
        model.load_decoder_checkpoint("external/lasso_checkpoints/best_model.pt")
        model.eval()
        return model.to(self.device)


    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        column_names = X.columns.tolist()

        if self.num_ensemble > 0:
            all_coeffs = []
            X_preprocessed, y_preprocessed = self.preprocessor.generate_preprocessed_datasets(X, y)
            y_p = y_preprocessed[0]
            for view_idx, X_p, in enumerate(X_preprocessed):
                X_p = X_p.squeeze(0).unsqueeze(1).to(self.device)# (seq_len, B, M) 
                train_size = int(X_p.shape[0] * 0.8)
                y_train = y_p.squeeze(0)[:train_size].unsqueeze(1).to(self.device)  
            
                with torch.no_grad():
                    coeffs = self.model(X_p, y_train)
                    coeffs = coeffs.squeeze(0)

                all_coeffs.append(coeffs)

            coef_stack = torch.stack(all_coeffs, dim=0) # (n_ensemble, n_features)
            mean_coef = coef_stack.mean(dim=0) # (n_features,)
            topk_idx = torch.topk(mean_coef, self.K).indices.tolist()
            self.selected_cols_ = [column_names[idx] for idx in topk_idx]
        else:
            train_size = int(X.shape[0] * 0.8)
            column_names = X.columns.tolist()

            X = torch.Tensor(X.values).unsqueeze(1).to(self.device) 
            y_train = torch.Tensor(y.iloc[:train_size].values).unsqueeze(1).to(self.device)  
            
            with torch.no_grad():
                coefficients = self.model(X, y_train)

            topk_idx = torch.topk(coefficients, self.K).indices.tolist()[0]
            self.selected_cols_ = [column_names[i] for i in topk_idx]

        del X, y_train, self.model
        memory_cleanup()
        log_mem()

        return self
