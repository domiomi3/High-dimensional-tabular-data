from tabrepo.models.utils import get_configs_generator_from_name as get_configs_generator_from_name_tr
# TODO: remove this and create a separate conf for TabPFNv2
from tabarena.models.utils import get_configs_generator_from_name

from autogluon.core.models import BaggedEnsembleModel

from high_tab.models.base import TabModel


class TabArenaModel(TabModel): 
    """
    TabArena/AutoGluon model for high-dimensional data preprocessing.
    Allows model-agnostic (pre-CV split) and model-specific (per-fold) FS/DR method use.
    By default creates a bagged ensemble of specificed base models.

    Parameters:
    - model_name (str): Name of the base model specified in TabArena.
    - model_class (class): Base model class as implemented in TabArena/Autogluon
    instantiated for model-specific preprocessing.
    - preprocessing (str): 'model-agnostic' or 'model-specific'.
    - task_type (str): 'binary', 'multiclass', or 'regression'.
    - eval_metric (str): Metric to be used for validation and test sets.
    - random_state (int): Random state for reproducibility.
    - device (str): Device used to train the model.
    - num_gpus (int): Number of GPUs used in training.
    - num_cpus (int): Number of CPUs used in training.
    - model_checkpoints_dir (str): Directory to save model checkpoints.
    - **kwargs: Contain HPs used in model-specific preprocessing under 'method_args' key.
    """
    def __init__(
        self,
        model_name: str,
        model_class,
        preprocessing: str,
        task_type: str,
        eval_metric: str,
        random_state: int,
        device: str,
        num_gpus: int,
        num_cpus: int,
        num_k_folds: int,
        model_checkpoints_dir: str,
        **kwargs,
    ):
        super().__init__(
            task_type=task_type,
            eval_metric=eval_metric,
            device=device,
            **kwargs,
        )

        self.model_name = model_name
        self.model_class = model_class
        self.preprocessing = preprocessing
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.num_k_folds = num_k_folds
        self.random_state = random_state
        self.cross_validation_bagging = True
        self.model_checkpoints_dir = model_checkpoints_dir
        self.kwargs = kwargs
     
    def fit(self, X, y):
        # get TabArena config (usually empty)
        if self.model_name=="TabPFNv2":
            meta = get_configs_generator_from_name_tr(self.model_name)
        else:
            meta = get_configs_generator_from_name(self.model_name)
        model_config = meta.manual_configs[0]
        if "hyperparameters" not in model_config:
            model_config["hyperparameters"] = {}
        if self.model_name in ["RealTabPFN-v2.5", "TabPFNv2"]:
            model_config["hyperparameters"]["ag.max_features"] = 10000 # allow >500 features                        

        if self.preprocessing=="model-specific":
            # create a base model class given the parent class (e.g. CatBoost, TabPFNv2Model)
            model_cls = self.model_class 
            # pass arguments for FS/DR method
            if self.kwargs["method_args"] and self.cross_validation_bagging:
                model_config["hyperparameters"].update(**self.kwargs)
        else: #TODO: fix this, only msp allowed
            model_cls = meta.model_cls
        
        base_config = {k: v for k, v in model_config.items() if k != "ag_args_ensemble"}

        if self.cross_validation_bagging:
            base_model = model_cls(
                problem_type=self.task_type,
                eval_metric=self.eval_metric,
                name=self.model_name,
                path=self.model_checkpoints_dir,
                **base_config
            )
            self.model = BaggedEnsembleModel(
                model_base=base_model, 
                random_state=self.random_state, 
                hyperparameters=dict(refit_folds=True),
                path=self.model_checkpoints_dir
            )
            self.model.params.update({
                "fold_fitting_strategy": "sequential_local", 
            })
            self.model = self.model.fit(X=X, y=y, k_fold=self.num_k_folds) # already stores the oof preds

            score = self.model.score_with_oof(y=y)
            self.val_score = -score if self.eval_metric in ["root_mean_squared_error", "log_loss"] else score # AG always maximizes
        else:
            self.model = model_cls(
                problem_type=self.task_type,
                eval_metric=self.eval_metric,
                name=self.model_name,
                path=self.model_checkpoints_dir,
                **model_config,
            )
            self.model.fit(X=X, y=y, num_gpus=self.num_gpus, num_cpus=self.num_cpus)
            self.val_score = None
        return self

    def predict(self, X):
        if self.task_type == "multiclass":
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def score(self, X, y):
        score = self.model.score(X, y)
        return -score if self.eval_metric in ["root_mean_squared_error", "log_loss"] else score # AG always maximizes
    