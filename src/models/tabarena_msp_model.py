import pandas as pd

from tabrepo.benchmark.models.ag.tabpfnv2.tabpfnv2_model import TabPFNV2Model
from autogluon.tabular.models.catboost.catboost_model import CatBoostModel

from preprocessors.tab_preprocessor import TabPreprocessor


class MSPModelMixin:
    """Mixin class that adds model-specific preprocessing to any TabArena/Autogluon model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimator = None

    def _get_model_params(self, msp_class=False):
        """Override to avoid unexpected arguments in the parent class."""
        params = super()._get_model_params()
        if not msp_class:
            if "method_args" in params:
                params.pop("method_args")
        return params

    def _preprocess(self, X: pd.DataFrame, is_train=False, y_train: pd.Series | None = None, **kwargs) -> pd.DataFrame: 
        """Override to allow custom FS/DR methods."""
        # allow to skip second call of preprocess() in super()._fit()
        if is_train and (self._estimator is not None):
            return X
        
        # default preprocessing 
        X = super()._preprocess(X, is_train=is_train, **kwargs)

        # extract hps for TabProcessor
        hps = self._get_model_params(msp_class=True)
        if "method_args" in hps:
            config = dict(hps["method_args"])
        else:
            raise RuntimeError("method_args not passed to the model.")

        # set up and run TabProcessor       
        random_state = config.pop("random_state", 44)
        if is_train:
            assert y_train is not None, "Preprocessing requires y_train when fitting"
            if self._estimator is None:
                self._estimator = TabPreprocessor(random_state, **config).fit(X, y_train)
        else:
            if self._estimator is None:
                raise RuntimeError("TabProcessor's estimator not fitted.")
        
        X = pd.DataFrame(self._estimator.transform(X))
        return X

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Override to allow preprocess() access y_train during cross-validation."""
        X = self.preprocess(X, y_train=y, is_train=True)
        super()._fit(X=X, y=y, **kwargs)

    def _preprocess_nonadaptive(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Override so it doesn't throw an error when self.features differ from X.columns."""
        return X

# classes inheriting from the original base model will have their methods overridden by the mixin class
class TabPFNV2MSPModel(MSPModelMixin, TabPFNV2Model):
    pass

class CatBoostMSPModel(MSPModelMixin, CatBoostModel):
    pass
