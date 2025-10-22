# models/__init__.py
from importlib import import_module

from high_tab.models.base import TabModel

_TABARENA_MODEL_REGISTRY = {
    "catboost_tab": ("CatBoost", "high_tab.models.tabarena_msp_model", "CatBoostMSPModel"),
    "tabpfnv2_tab": ("TabPFNv2", "high_tab.models.tabarena_msp_model", "TabPFNV2MSPModel"),
    "tabpfn_wide": ("TabPFNv2", "high_tab.models.tabarena_msp_model", "TabPFNWideMSPModel"),
}

def make_model(model_key: str, **kwargs) -> "TabModel":
    model_key = model_key.lower()
    if model_key == "tabpfnv2_org":
        return import_module("high_tab.models.tabpfnv2_original").TabPFNv2Original(**kwargs)

    if model_key in _TABARENA_MODEL_REGISTRY:
        tabarena_name, model, model_class = _TABARENA_MODEL_REGISTRY[model_key]
        tabarena_class = getattr(import_module(model), model_class)
        TabArenaModel = getattr(import_module("high_tab.models.tabarena_model"), "TabArenaModel")
        return TabArenaModel(model_name=tabarena_name, model_class=tabarena_class, **kwargs)

    raise ValueError(f"Unknown model_key '{model_key}'. Known: {list(_TABARENA_MODEL_REGISTRY) + ['tabpfnv2_org']}")