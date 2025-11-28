from autogluon.tabular.models.tabpfnv2.tabpfnv2_model import TabPFNV2Model, FixedSafePowerTransformer # _patch_local_kdi_transformer
from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model

from high_tab.preprocessors.tab_preprocessor import TabPreprocessor


class _MSPBaseMixin:
    """Mixin base class that adds model-specific preprocessing to any TabArena/Autogluon model."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimator = None
        
    def _get_model_params(self, msp_class=False):
        """Override to avoid unexpected arguments in the parent class."""
        params = super()._get_model_params()
        if not msp_class and "method_args" in params:
            params.pop("method_args")
        return params

    def _preprocess(self, X, is_train=False, y_train=None, **kwargs):
        """
        Override to allow custom FS/DR methods.
        is_train defaulting to False allows the models to only transform the X 
        during inference. Requires the parent's fit() to call preprocess() with 
        is_train=True arguments. 
        """
        import pandas as pd

        # skip second call during super()._fit()
        if is_train and (self._estimator is not None):
            return X

        # default preprocessing 
        X_sup_preprocessed = super()._preprocess(X, is_train=is_train, **kwargs)

        # extract hps for TabProcessor
        hps = self._get_model_params(msp_class=True)
        if "method_args" not in hps:
            raise RuntimeError("method_args not passed to the model.")
        config = dict(hps["method_args"])
        random_state = config.pop("random_state", 44)
        
        if is_train:
            assert y_train is not None, "Preprocessing requires y_train when fitting"
            if self._estimator is None:
                self._estimator = TabPreprocessor(random_state, **config).fit(X_sup_preprocessed, y_train)
        else:
            if self._estimator is None:
                raise RuntimeError("TabProcessor's estimator not fitted.")
        X_preprocessed = pd.DataFrame(self._estimator.transform(X_sup_preprocessed))
        return X_preprocessed

    # template method: do not override this in subclasses
    def _fit(self, X, y, **kwargs):
        """Override to allow preprocess() access y_train during cross-validation."""
        X_preprocessed = self.preprocess(X, y_train=y, is_train=True)
        return self._custom_fit(X_preprocessed, y, **kwargs)

    # hook to override in variants
    def _custom_fit(self, X, y, **kwargs):
        return super()._fit(X=X, y=y, **kwargs)

    def _preprocess_nonadaptive(self, X, **kwargs):
        """Override so it doesn't throw an error when self.features differ from X.columns."""
        return X

    #TODO: how to do regression?
    # def _predict_proba(self, X ,y, **kwargs):
    #     X_preprocessed = self.preprocess(X, is_train=False)
    #     return super()._predict_proba(X=X_preprocessed, y=y, **kwargs)
    
# uses the default hook
class MSPDefaultModelMixin(_MSPBaseMixin):
    pass

# overrides the _fit_custom hook for TabPFN-Wide
class MSPWideModelMixin(_MSPBaseMixin):
    def _custom_fit(self, X, y, **kwargs):
        """Override to load a pretrained TabPFN-Wide model."""
        import torch 
        import types

        from torch.cuda import is_available
        from tabpfn.model.loading import load_model_criterion_config, resolve_model_path
        from tabpfn import TabPFNClassifier
        from tabpfn.model import preprocessing

        from tabpfnwide.patches import fit

        preprocessing.SafePowerTransformer = FixedSafePowerTransformer


        # msp preprocessing
        X = self.preprocess(X, y_train=y, is_train=True)
        # _patch_local_kdi_transformer()

        is_classification = self.problem_type in ["binary", "multiclass"]
        if not is_classification: 
            raise ValueError(f"TabPFN-Wide only supports classification tasks.")
        model_base = TabPFNClassifier

        num_gpus = 1 #TODO: pass as class attribute
        num_cpus = 1
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        X = self.preprocess(X, is_train=True)

        hps = self._get_model_params()
        hps["device"] = device
        hps["n_jobs"] = num_cpus
        hps["categorical_features_indices"] = self._cat_indices

        _, model_dir, _, _ = resolve_model_path(
            model_path=None,
            which="classifier" if is_classification else "regressor",
        )
        if is_classification:
            if "classification_model_path" in hps:
                hps["model_path"] = model_dir / hps.pop("classification_model_path")
            if "regression_model_path" in hps:
                del hps["regression_model_path"]
       
        # Resolve inference_config
        inference_config = {
            _k: v
            for k, v in hps.items()
            if k.startswith("inference_config/") and (_k := k.split("/")[-1])
        }
        if inference_config:
            hps["inference_config"] = inference_config
        for k in list(hps.keys()):
            if k.startswith("inference_config/"):
                del hps[k]

        # TODO: remove power from search space and TabPFNv2 codebase
        # Power transform can fail. To avoid this, make all power be safepower instead.
        if "PREPROCESS_TRANSFORMS" in inference_config:
            safe_config = []
            for preprocessing_dict in inference_config["PREPROCESS_TRANSFORMS"]:
                if preprocessing_dict["name"] == "power":
                    preprocessing_dict["name"] = "safepower"
                safe_config.append(preprocessing_dict)
            inference_config["PREPROCESS_TRANSFORMS"] = safe_config
        if "REGRESSION_Y_PREPROCESS_TRANSFORMS" in inference_config:
            safe_config = []
            for preprocessing_name in inference_config[
                "REGRESSION_Y_PREPROCESS_TRANSFORMS"
            ]:
                if preprocessing_name == "power":
                    preprocessing_name = "safepower"
                safe_config.append(preprocessing_name)
            inference_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = safe_config

        # Resolve model_type
        n_ensemble_repeats = hps.pop("n_ensemble_repeats", None)
        _ = hps.pop("model_type", "no") == "dt_pfn"
        
        if n_ensemble_repeats is not None:
            hps["n_estimators"] = n_ensemble_repeats
        
        # TODO: find a more elegant way
        msp_hps = self._get_model_params(msp_class=True)
        if "method_args" not in msp_hps:
            raise RuntimeError("method_args not passed to the model.")
        config = dict(msp_hps["method_args"])
        num_estimators = config.pop("num_estimators", 8)
        
        # load the weights of the pretrained model (not fitted)
        #TODO: make it an attribute
        checkpoint_path = f"external/tabpfnwide/models/TabPFN-Wide-8k_submission.pt"

        wide_model = load_model_criterion_config(
                model_path=None,
                check_bar_distribution_criterion=False,
                cache_trainset_representation=False,
                which='classifier',
                version='v2',
                download_if_not_exists=True,
        )[0][0]
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        wide_model.load_state_dict(checkpoint)
        wide_model.to(device)
        
        self.model = model_base()
        self.model = self.model.create_default_for_version(version="v2", n_estimators=num_estimators, **hps)
        
        if is_classification and not hasattr(self.model, 'softmax_temperature_'):
            self.model.softmax_temperature_ = self.model.softmax_temperature
            self.model.tuned_classification_thresholds_ = None
                           
        self.model.fit = types.MethodType(fit, self.model)
        self.model.fit(X, y, model=wide_model)

# classes inheriting from the original base model will have their methods overridden by the mixin class
class RealTabPFNv25ModelMSPModel(MSPDefaultModelMixin, RealTabPFNv25Model):
    pass

class TabPFNV2MSPModel(MSPDefaultModelMixin, TabPFNV2Model):
    pass

class TabPFNWideMSPModel(MSPWideModelMixin, TabPFNV2Model):
    pass

class CatBoostMSPModel(MSPDefaultModelMixin, CatBoostModel):
    pass


