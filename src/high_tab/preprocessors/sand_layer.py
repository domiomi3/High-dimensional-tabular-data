import torch
import logging

import pandas as pd
import numpy as np

from torch.optim import Adam
from torch.utils.data import DataLoader

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.utils import meta_dataset_collator

from high_tab.utils.hardware import memory_cleanup

logger = logging.getLogger(__name__)

class SANDProcessor:
    """
    Trains a SAND layer for preprocessed inputs with TabPFN gradient signal.
    For preprocessing, use selected_columns_ (similar to random_fs method).
    """
    def __init__(self, num_features=150, model_type="tabpfnv2_tab", random_state=44, 
                 device=None, finetune_epochs=5, n_estimators_ft=1, task_type=None, 
                 sigma=1.5, lr=1.5e-6, val_ratio=0.3, meta_batch_size=1,
                 max_data_size=256):
        self.K = int(num_features)
        self.model_type = model_type
        self.wide_model = None
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = finetune_epochs
        self.n_estimators_ft = n_estimators_ft
        self.task_type = task_type
        self.sigma = sigma
        self.lr = lr
        self.val_ratio = val_ratio
        self.meta_batch_size = meta_batch_size
        self.max_data_size = max_data_size
        self.selected_cols_ = None
        self.classification_task = False if self.task_type == "regression" else True
    
    class _SAND(torch.nn.Module):
        """
        PyTorch SAND layer implementation.
        """
        def __init__(self, d, k, sigma):
            super().__init__()
            self.k, self.sigma = k, sigma
            init = (k / d) ** 0.5
            self.w_raw = torch.nn.Parameter(torch.full((d,), init, dtype=torch.float32))
        @staticmethod
        def _project(w_raw, k, eps=1e-12):
            w = torch.clamp(w_raw.abs(), max=1.0) #clip to [0,1]
            return w / (w.pow(2).sum().sqrt() + eps) * (k ** 0.5)
        def forward(self, x, training=False):
            wn = self._project(self.w_raw, self.k)
            y = x * wn
            if training:
                y = y + torch.randn_like(y) * (self.sigma * (1.0 - wn.abs()))
            return y
        @torch.no_grad()
        def topk(self):
            wn = self._project(self.w_raw, self.k)
            return torch.topk(wn.abs(), self.k).indices
    
    @staticmethod
    def _to_tabpfn_arrays(X, y, task_type):
        if isinstance(X, pd.DataFrame):
            X_np = X.to_numpy(dtype=np.float16)
        else:
            X_np = np.asarray(X, dtype=np.float16)
        if isinstance(y, pd.Series):
            y_np = y.to_numpy()
        else:
            y_np = np.asarray(y)
        if task_type == "regression":
            y_np = y_np.astype(np.float16)
        else:
            # ensure integer class ids
            _, y_np = np.unique(y_np, return_inverse=True)
            y_np = y_np.astype(np.int64)
        return X_np, y_np


    def _setup_tabpfn_model(self):
        # for feature dim preserving
        inference_config = {
            "PREPROCESS_TRANSFORMS": [
                {
                    "name": "none",                 # or "robust", "safepower" if you want scaling
                    "categorical_name": "numeric",  # no onehot / ordinal expansion
                    "append_original": False,       # never append engineered cols
                    "subsample_features": -1,       # -1 = no subsampling
                    "global_transformer_name": None,# no SVD
                    "differentiable": False,
                }
            ],
            "FEATURE_SHIFT_METHOD": None,
            "CLASS_SHIFT_METHOD": None,
            "FINGERPRINT_FEATURE": False,
            "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": 0,
            "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0,
            "SUBSAMPLE_SAMPLES": None,
            "POLYNOMIAL_FEATURES": "no",
        }
        model_class = TabPFNClassifier if self.classification_task else TabPFNRegressor
        model_config = dict(
            ignore_pretraining_limits=True,
            device=self.device,
            n_estimators=self.n_estimators_ft,
            random_state=self.random_state,
            inference_precision=torch.float16,
            inference_config=inference_config,
        )
        predictor = model_class(**model_config, fit_mode="batched", differentiable_input=False)
        if self.classification_task:
            predictor._initialize_model_variables()
        
        if self.model_type=="tabpfn_wide":
            import types
            from tabpfn.model.loading import load_model_criterion_config, resolve_model_path
            from tabpfnwide.patches import fit, fit_from_preprocessed
            
            checkpoint_path = f"external/tabpfnwide/models/TabPFN-Wide-8k_submission.pt"
            wide_model, _, _ = load_model_criterion_config(
                model_path=None,
                check_bar_distribution_criterion=False,
                cache_trainset_representation=False,
                which='classifier',
                version='v2',
                download=True,
            )
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            wide_model.load_state_dict(checkpoint)
            wide_model.to(self.device)
            self.wide_model = wide_model

            predictor.fit = types.MethodType(fit, predictor)
            predictor.fit_from_preprocessed = types.MethodType(fit_from_preprocessed, predictor)

        return predictor


    @staticmethod
    def _make_stable_splitter(test_size, random_state, use_stratify: bool, eps=1e-6):
        """
        Returns a spliting function used inside the get_preprocessed_datasets()
        on chunks created with _get_stratified_sets().
        Ensures stratified train/test splits and same feature dimensionality between
        chunks.
        """
        from sklearn.model_selection import train_test_split

        def split_fn(X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y)
            # decide stratification per-chunk
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y if use_stratify else None
                )
            except ValueError:
                # fallback if a small chunk makes stratify invalid
                logger.info("Could not stratify the dataset chunk.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=None
                )
            # replace non-finite vals with 0
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            
            # break zero-variance cols on train so nothing gets dropped in get_preprocessed_dataset()
            std = X_train.std(axis=0)
            zero_var = std == 0.0
            if zero_var.any():
                rng = np.random.default_rng(random_state)
                jitter = rng.normal(scale=eps, size=(X_train.shape[0], int(zero_var.sum()))).astype(np.float32)
                X_train[:, zero_var] += jitter
            return X_train, X_test, y_train, y_test

        return split_fn

    @staticmethod
    def _get_stratified_sets(X,y,max_data_size,random_state, use_stratify):
        """
        Returns list of (X,y) tuples corresponding to a stratified fold.
        This way for classification tasks each training set is guaranteed to contain all classes
        before it is passed to get_preprocessed_datasets().  
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        import math

        n_splits = max(2, int(math.ceil(len(y) / max(1, max_data_size)))) # number of chunks based on max_data_size

        if use_stratify: # classification
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            fold_iter = skf.split(np.zeros(len(y)), y)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            fold_iter = kf.split(np.zeros(len(y)))

        training_datasets = []
        for _, fold_idx in fold_iter: # each test_idx is one fold
            X_fold, y_fold = X[fold_idx], y[fold_idx]
            training_datasets.append((X_fold, y_fold))
    
        return training_datasets

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """SAND layer training for FS with frozen TabPFN predictor (classifier or regressor)."""

        predictor = self._setup_tabpfn_model() 

        X_np, y_np = self._to_tabpfn_arrays(X, y, self.task_type) # to numpy arrays
        
        # stable splitter for no feature dim mismatch problem
        splitter = SANDProcessor._make_stable_splitter(
            test_size=self.val_ratio,
            random_state=self.random_state,
            use_stratify=self.classification_task,
        )

        # workaround for issue with non-stratified chunks when using max_data_size in get_preprocessed_datasets
        training_chunks = SANDProcessor._get_stratified_sets(
            X_np, y_np, max_data_size=self.max_data_size, random_state=self.random_state, 
            use_stratify=self.classification_task
        )
        training_datasets = [
            predictor.get_preprocessed_datasets(
                X_i, y_i, splitter, max_data_size=None
            )[0] for (X_i, y_i) in training_chunks
        ]
        finetuning_dataloader = DataLoader(
            training_datasets,
            batch_size=self.meta_batch_size,
            collate_fn=meta_dataset_collator,
        )

        sand_model, optimizer = None, None # lazy init
        loss_log = []
        grad_norm_log = []

        logger.info(f"Finetuning SAND layer for {self.epochs} epochs")
        for epoch in range(self.epochs):
            epoch_loss_sum = 0.0
            epoch_grad_sum = 0.0
            epoch_batches = 0

            with torch.amp.autocast(dtype=torch.float16, device_type=self.device):
                for data_batch in finetuning_dataloader:
                    if self.classification_task:
                        (X_trains_preprocessed,
                        X_tests_preprocessed,
                        y_trains_preprocessed,
                        y_tests_preprocessed,
                        cat_ixs,
                        confs) = data_batch

                        if len(np.unique(y_trains_preprocessed)) != len(np.unique(y_tests_preprocessed)):
                            continue  # Skip batch if splits don't have all classes
                    else:
                        (
                            X_trains_preprocessed,
                            X_tests_preprocessed,
                            y_trains_preprocessed,
                            y_tests_preprocessed,
                            cat_ixs,
                            confs,
                            raw_bd,
                            znorm_bd,
                            _,
                            _
                        ) = data_batch
                        predictor.raw_space_bardist_ = raw_bd[0]
                        predictor.bardist_ = znorm_bd[0]

                    # lazy init SAND
                    if sand_model is None:
                        feat_dim = X_trains_preprocessed[0].shape[-1]
                        sand_model = self._SAND(d=feat_dim, k=self.K, sigma=self.sigma).to(self.device)
                        optimizer = Adam(sand_model.parameters(), lr=self.lr)

                    optimizer.zero_grad(set_to_none=True)

                    # apply SAND
                    X_train_sand = [sand_model(x.to(self.device), training=True) for x in X_trains_preprocessed]
                    X_test_sand = [sand_model(x.to(self.device), training=False) for x in X_tests_preprocessed]

                    with torch.no_grad():
                        fit_kwargs = {'no_refit': True}
                        if self.model_type == 'tabpfn_wide':
                            fit_kwargs['model'] = self.wide_model # for tabpfn_wide

                        predictor.fit_from_preprocessed(
                            X_train_sand, y_trains_preprocessed, cat_ixs, confs, **fit_kwargs
                        )
                    
                    # freeze model
                    for p in predictor.model_.parameters(): p.requires_grad = False
                    predictor.model_.eval()

                    if self.classification_task:
                        logits = predictor.forward(X_test_sand, return_logits=True)
                        loss = torch.nn.CrossEntropyLoss()(logits, y_tests_preprocessed.to(self.device))
                    else:
                        preds, _, _ = predictor.forward(X_test_sand)
                        loss = znorm_bd[0](preds, y_tests_preprocessed.to(self.device)).mean()
                    loss.backward()

                    # grad norm (SAND only)
                    g = sand_model.w_raw.grad
                    if g is not None:
                        epoch_grad_sum += float(g.detach().norm().item())

                    optimizer.step()

                    epoch_loss_sum += float(loss.detach().item())
                    epoch_batches += 1

            # per-epoch summary
            if epoch_batches == 0:
                logger.info(f"[epoch {epoch}] no valid batches; logging NaN")
                loss_log.append(float("nan"))
                grad_norm_log.append(float("nan"))
            else:
                avg_loss = epoch_loss_sum / epoch_batches
                avg_grad = epoch_grad_sum / epoch_batches
                loss_log.append(avg_loss)
                grad_norm_log.append(avg_grad)
                logger.info(f"[epoch {epoch}] loss={avg_loss:.4f}  grad_norm={avg_grad:.3e}")

        # finalize weights, extract top k columns
        with torch.no_grad():
            weights = SANDProcessor._SAND._project(sand_model.w_raw, sand_model.k).abs().cpu().numpy()

        topk = np.argsort(-weights)[: self.K]
        self.selected_cols_ = list(X.columns[topk]) if hasattr(X, "columns") else topk.tolist()

        # keep logs for later inspection
        self._epoch_loss = loss_log
        self._epoch_grad_norm = grad_norm_log

        # clean up memory
        del finetuning_dataloader, training_datasets, training_chunks, optimizer, X_np, y_np
        memory_cleanup()

        return self
