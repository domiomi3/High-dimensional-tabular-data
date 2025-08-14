from tabrepo.models.utils import get_configs_generator_from_name
from autogluon.core.models import BaggedEnsembleModel
from autogluon.core.metrics import get_metric

from models.base import TabModel
from utils.hardware import count_gpus


class TabArenaModel(TabModel): # TODO: add ensemble bagging, CPU training
    """Select a model to run, which we automatically load in the code below.

    Note: not all models are available for all task types.

    The recommended options are:
        - "RealMLP"
        - "TabM"
        - "LightGBM"
        - "CatBoost"
        - "XGBoost"
        - "ModernNCA"
        - "TabPFNv2"
        - "TabICL"
        - "TorchMLP"
        - "TabDPT"
        - "EBM"
        - "FastaiMLP"
        - "ExtraTrees
        - "RandomForest"
        - "KNN"
        - "Linear"

    You can also import it manually from TabArena / AutoGluon, which we recommend
    for practical applications, for example:
    - RealMLP: from tabrepo.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel
    - Catboost: from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
    """
    def __init__(
        self,
        model_name: str,
        task_type: str,
        eval_metric,
        eval_func,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            task_type=task_type,
            eval_metric=eval_metric,
            eval_func=eval_func,
            device=device,
            **kwargs,
        )

        self.model_name = model_name
        self.kwargs = kwargs
        self.cross_validation_bagging = True
     
    def fit(self, X, y):
        meta = get_configs_generator_from_name(self.model_name)
        model_cls = meta.model_cls
        model_config = meta.manual_configs[0]
        metric = "root_mean_squared_error" if isinstance(self.eval_metric[0], str) and self.eval_metric[0].lower() in {"rmse", "root_mean_squared_error"} else self.eval_metric[0]
        num_gpus = count_gpus() if self.device == "cuda" else 0
        num_cpus = 1
        if self.cross_validation_bagging and self.model_name != "TabPFNv2": # add logging for bagging 
            #TODO: there's data leakage so even oof scores might be optimistic 
            self.model = BaggedEnsembleModel(model_cls(problem_type=self.task_type, eval_metric=metric, name=self.model_name, **model_config))
            self.model.params["fold_fitting_strategy"] = "sequential_local"
            self.model.fit(X=X, y=y, k_fold=8)
            score = self.model.score_with_oof(y=y)
            self.val_score = -score if metric == "root_mean_squared_error" else score # AG always maximizes
        else:
            self.model = model_cls(
                problem_type=self.task_type,
                eval_metric=metric,
                name=self.model_name,
                path="models", #TODO??
                **self.kwargs,
            )
            self.model.fit(X=X, y=y, num_gpus=num_gpus, num_cpus=num_cpus)
            self.val_score = None
        return self

    def predict(self, X):
        if self.task_type == "multiclass":
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def score(self, X, y):
        return self.eval_func(y, self.predict(X))
    