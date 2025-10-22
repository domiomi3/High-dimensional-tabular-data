from typing import Any

from high_tab.models.base import TabModel

class TabPFNv2Original(TabModel):
    """
    TabPFNv2 paper implementation.
    """
    def __init__(
        self,
        task_type: str,
        eval_metric: Any,
        eval_func: Any,
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
        self.model_name = "TabPFNv2"  
        self.kwargs = kwargs                    

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor, TabPFNClassifier

        if self.task_type == "regression":
            self.model = TabPFNRegressor(
                ignore_pretraining_limits=self.kwargs.get("ignore_limits", False),
                device=self.device,
            )
        else:
            self.model = TabPFNClassifier(
                ignore_pretraining_limits=self.kwargs.get("ignore_limits", False),
                device=self.device,
            )
        self.model.fit(X, y)
        self.val_score = None
        return self

    def predict(self, X):
        if self.task_type == "multiclass":
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def score(self, X, y):
        return self.eval_func(y, self.predict(X))