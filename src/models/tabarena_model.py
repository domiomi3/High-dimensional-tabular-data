from models.base import TabModel
from utils.hardware import count_gpus


class TabArenaModel(TabModel):
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
     
    def fit(self, X, y):
        from tabrepo.models.utils import \
        get_configs_generator_from_name

        meta = get_configs_generator_from_name(self.model_name)
        model_cls = meta.model_cls
        metric = ("root_mean_squared_error"
                    if "rmse" in self.eval_metric else self.eval_metric[0])
        num_gpus = count_gpus() if self.device == "cuda" else 0
        num_cpus = 1
        self.model = model_cls(
            problem_type=self.task_type,
            eval_metric=metric,
            name=self.model_name,
            path="models", #TODO??
            **self.kwargs,
        ).fit(X=X, y=y, num_gpus=num_gpus, num_cpus=num_cpus)
        return self

    def predict(self, X):
        if self.task_type == "multiclass":
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def score(self, X, y):
        return self.eval_func(y, self.predict(X))
    