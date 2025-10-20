from .base import TabModel
from .tabpfnv2_original import TabPFNv2Original
from .tabarena_model import TabArenaModel  
from .tabarena_msp_model import TabPFNV2MSPModel, CatBoostMSPModel, TabPFNWideMSPModel

_TABARENA_MODEL_REGISTRY = {
    # "realmlp_tab":   "RealMLP",
    # "tabm_tab":      "TabM",
    "catboost_tab": {
        "name": "CatBoost",
        "class": CatBoostMSPModel
    },
    "tabpfnv2_tab": {
        "name": "TabPFNv2", 
        "class": TabPFNV2MSPModel
    },
    "tabpfn_wide": {
        "name": "TabPFNv2", # needs to be recognizable by TabArena
        "class": TabPFNWideMSPModel
    }
    # "xgboost_tab":   "XGBoost",
    # "modernnca_tab": "ModernNCA",
    # "tabicl_tab":    "TabICL",
    # "torchmlp_tab":  "TorchMLP",
    # "tabdpt_tab":    "TabDPT",
    # "ebm_tab":       "EBM",
    # "fastaimlp_tab": "FastaiMLP",
    # "extratrees_tab":"ExtraTrees",
    # "randomforest_tab": "RandomForest",
    # "knn_tab":       "KNN",
    # "linear_tab":    "Linear",
}

def make_model(model_key: str, **kwargs) -> TabModel:
    """Return an instantiated TabModel according to `model_key`."""
    model_key = model_key.lower()

    # the one non-TabArena special case
    if model_key == "tabpfnv2_org":
        return TabPFNv2Original(**kwargs)

    # TabArena models 
    if model_key in _TABARENA_MODEL_REGISTRY:
        tabarena_name = _TABARENA_MODEL_REGISTRY[model_key]["name"]
        tabarena_class = _TABARENA_MODEL_REGISTRY[model_key]["class"]
        return TabArenaModel(model_name=tabarena_name, model_class=tabarena_class, **kwargs)

    raise ValueError(
        f"Unknown model_key '{model_key}'. "
        f"Known TabArena keys: {list(_TABARENA_MODEL_REGISTRY)}, "
        f"and 'tabpfnv2_org' for the original TabPFNv2 implementation."
    )