from .base import TabModel
from .tabpfnv2_original import TabPFNv2Original
from .tabarena_model import TabArenaModel  

_TABARENA_NAME_MAP: dict[str, str] = {
    "realmlp_tab":   "RealMLP",
    "tabm_tab":      "TabM",
    "catboost_tab":  "CatBoost",
    "xgboost_tab":   "XGBoost",
    "modernnca_tab": "ModernNCA",
    "tabpfnv2_tab":  "TabPFNv2",
    "tabicl_tab":    "TabICL",
    "torchmlp_tab":  "TorchMLP",
    "tabdpt_tab":    "TabDPT",
    "ebm_tab":       "EBM",
    "fastaimlp_tab": "FastaiMLP",
    "extratrees_tab":"ExtraTrees",
    "randomforest_tab": "RandomForest",
    "knn_tab":       "KNN",
    "linear_tab":    "Linear",
}

def make_model(model_key: str, **kwargs) -> TabModel:
    """Return an instantiated TabModel according to `model_key`."""
    model_key = model_key.lower()

    # a) the one non-TabArena special case
    if model_key == "tabpfnv2_org":
        return TabPFNv2Original(**kwargs)

    # b) all TabArena models share the same concrete class
    if model_key in _TABARENA_NAME_MAP:
        tabarena_name = _TABARENA_NAME_MAP[model_key]
        return TabArenaModel(model_name=tabarena_name, **kwargs)

    raise ValueError(
        f"Unknown model_key '{model_key}'. "
        f"Known TabArena keys: {list(_TABARENA_NAME_MAP)}, "
        f"plus 'tabpfnv2_org' for the original implementation."
    )