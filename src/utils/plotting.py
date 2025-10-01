import os
import pandas as pd
import hashlib
import seaborn as sns

from matplotlib.colors import to_hex


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
                    val_metric = [m_col for m_col in df_all.columns if m_col == f"val_{metric}"][0]
                    columns_req = columns + [metric] + [val_metric]
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

def build_model_color_map(display_name_map, palette="colorblind", overrides=None):
    """
    Create a stable mapping: model_key -> hex color using a Seaborn palette.

    Stability comes from the insertion order of `display_name_map` (Python 3.7+ preserves it).
    If you always define the models in the same order, colors stay fixed across runs.

    Args:
        display_name_map (dict): {model_key: display_name}. Keys define color order.
        palette (str | list): Seaborn palette name or color list.
        overrides (dict | None): Optional {model_key: "#RRGGBB"} to force specific colors.

    Returns:
        dict: {model_key: "#RRGGBB"}
    """
    overrides = overrides or {}
    keys = list(display_name_map.keys())
    pal = sns.color_palette(palette, n_colors=len(keys))

    color_map = {}
    j = 0
    for i, k in enumerate(keys):
        if k in overrides:
            color_map[k] = overrides[k]
        else:
            # If overrides reduced available palette positions, keep assigning sequentially
            color_map[k] = to_hex(pal[j % len(pal)])
            j += 1
    return color_map


def color_for_model(model_key, base_color_map, palette="colorblind", cycle_len=12):
    """
    Deterministic color for models not present in `base_color_map`.
    Uses MD5 of the key to pick a stable index in a small palette.

    Args:
        model_key (str): The model identifier.
        base_color_map (dict): Known {model_key: "#RRGGBB"}.
        palette (str | list): Seaborn palette name or color list for fallback.
        cycle_len (int): Size of the fallback palette.

    Returns:
        str: "#RRGGBB" color.
    """
    if model_key in base_color_map:
        return base_color_map[model_key]
    pal = sns.color_palette(palette, n_colors=cycle_len)
    idx = int(hashlib.md5(model_key.encode("utf-8")).hexdigest(), 16) % cycle_len
    return to_hex(pal[idx])