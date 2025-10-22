from abc import ABC, abstractmethod
from typing import Any, Callable


class TabModel(ABC):
    """Common interface for all tabular models."""
    def __init__(
        self,
        task_type: str,
        eval_metric: str | list[str],
        device: str,
        **kwargs: Any,
    ) -> None:
        self.task_type   = task_type
        self.eval_metric = eval_metric
        self.device  = device
        self.kwargs      = kwargs        

    @abstractmethod
    def fit(self, X, y) -> "TabModel": ...
    @abstractmethod
    def predict(self, X): ...
    @abstractmethod
    def score(self, X, y):
        """Returns a single scalar score using self.eval_func."""
