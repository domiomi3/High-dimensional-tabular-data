import os
import pandas as pd
import hashlib
import seaborn as sns
from typing import Dict, Optional
from matplotlib.colors import to_hex


DISPLAY_NAME_MAP = {
    "tabpfnv2_tab_map": "TabPFNv2 (MAP)",
    "tabpfnv2_tab": "TabPFNv2",
    "tabpfnv2_org": "TabPFNv2 Original",
    "catboost_tab_cpu": "CatBoost (8 CPU)",
    "catboost_tab": "CatBoost",
    "tabarena_sota": "TabArena SOTA",
    "tabpfn_wide": "TabPFN-Wide",
}
DEFAULT_PALETTE = "colorblind"
DEFAULT_EXTRA_POOL = 10


def load_results(rootdir, requested_columns=[]):
    columns = ["dataset_name", "method", "model", "repeat", "fold"] + requested_columns
    metrics=["log_loss", "roc_auc", "rmse"]
    df_list = []

    for subdir, _, files in os.walk(rootdir):
        if any([filename.endswith('.csv') for filename in files]):
            for filename in files:
                if filename.endswith('.csv'):
                    csv_path = os.path.join(subdir, filename)
                    df_all = pd.read_csv(csv_path)
                    metric = [m_col for m_col in df_all.columns if m_col in metrics][0]
                    # val_metric = [m_col for m_col in df_all.columns if m_col == f"val_{metric}"][0]
                    columns_req = columns + [metric] 
                    # + [val_metric]
                    df_req = df_all[columns_req].copy()
                    df_list.append(df_req)

    return df_list, metric


def get_mean_std(df_list, metric, if_val=True):
    perf_dict = {}
    for df_single in df_list:
        method = df_single["method"][0]            
        perf_dict[method] = {}
        perf_dict[method]["mean"] = df_single[metric].mean()
        perf_dict[method]["std"] = df_single[metric].std()
        
        if f"val_{metric}" in df_single.columns and if_val:
            perf_dict[method]["val_mean"] = df_single[f"val_{metric}"].mean()
            perf_dict[method]["val_std"] = df_single[f"val_{metric}"].std()
    return perf_dict


def merge_results(root_dirs):
    comb_results = {}
    for root_dir in root_dirs:
        results, metric = load_results(rootdir=root_dir)
        model = results[0]["model"][0]
        dataset = results[0]["dataset_name"][0]
        avg_results = get_mean_std(results, metric)
        comb_results[model] = avg_results
    
    return comb_results, dataset, metric

# --- Display-name helper ------------------------------------------------------

def get_display_name(model_key: str, display_name_map: Optional[Dict[str, str]] = None) -> str:
    """
    Return a human-friendly name for a model key. Falls back to key if unknown.
    """
    base = display_name_map or DISPLAY_NAME_MAP
    return base.get(model_key, model_key)


# --- Color mapping: fixed + deterministic fallback ---------------------------

def build_model_color_map(
    display_name_map: Optional[Dict[str, str]] = None,
    palette=DEFAULT_PALETTE,
    overrides: Optional[Dict[str, str]] = None,
    extra_pool: int = DEFAULT_EXTRA_POOL,
) -> Dict[str, str]:
    """
    Build a FIXED color map for known models (keys from display_name_map),
    drawing from ONE Seaborn palette. These colors never change across plots.

    Unknown models are NOT added here—use color_for_model() for those.

    Args:
        display_name_map: {model_key: display_name}. If None, uses DISPLAY_NAME_MAP.
        palette: Seaborn palette name or explicit color list.
        overrides: Optional {model_key: "#RRGGBB"} to force specific colors.
        extra_pool: Also allocate this many additional palette colors so unknown
                    models can be colored from the SAME palette without collision.

    Returns:
        {model_key: "#RRGGBB"} for the fixed base models only.
    """
    overrides = overrides or {}
    base = display_name_map or DISPLAY_NAME_MAP

    base_keys = list(base.keys())
    # One palette for both fixed + future colors (ensures same scheme).
    pal = sns.color_palette(palette, n_colors=len(base_keys) + max(0, extra_pool))

    color_map: Dict[str, str] = {}
    for i, k in enumerate(base_keys):
        color_map[k] = overrides.get(k, to_hex(pal[i]))
    return color_map


def color_for_model(
    model_key: str,
    base_color_map: Dict[str, str],
    palette=DEFAULT_PALETTE,
    extra_pool: int = DEFAULT_EXTRA_POOL,
) -> str:
    """
    Stable color for ANY model key.

    - If the key is in base_color_map, return its fixed color.
    - Otherwise, pick a deterministic color from the SAME palette,
      using an MD5 hash, ensuring no collision with fixed colors.

    Args:
        model_key: The model identifier.
        base_color_map: Fixed colors returned by build_model_color_map().
        palette: Seaborn palette name or explicit color list (must match the one used above).
        extra_pool: Size of the fallback pool; should be >= number of possible unknown models.

    Returns:
        "#RRGGBB" color string.
    """
    if model_key in base_color_map:
        return base_color_map[model_key]

    fixed_n = len(base_color_map)
    pal = sns.color_palette(palette, n_colors=fixed_n + max(1, extra_pool))

    # Deterministic index into the fallback segment [fixed_n, fixed_n + extra_pool)
    idx = int(hashlib.md5(model_key.encode("utf-8")).hexdigest(), 16) % max(1, extra_pool)
    return to_hex(pal[fixed_n + idx])
