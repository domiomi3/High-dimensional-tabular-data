import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from datetime import datetime

sys.path.append(".")
from utils import *


def plot_results(data_path, fig_dir, fig_path, metric, manual_df, title, method_name_map):
    os.makedirs(fig_dir, exist_ok=True)
    fig_abs_path = os.path.join(fig_dir, fig_path)

    base_color = "#a6d8af"
    manual_color = "#f4a6a6"
    base_e_color = "#3e7949"
    manual_e_color = "#c96a6a"

    all_files = glob.glob(data_path)

    if not all_files:
        print(f"[WARNING] No files matched: {data_path}")
        return

    dfs = []
    hp_values = None  # to store HPs from the first file

    for file in all_files:
        df = pd.read_csv(file)
        method_name = df["method"].iloc[0]
        df["method"] = method_name
        dfs.append(df)

        if hp_values is None:
            hp_values = extract_hyperparams_from_filename(file)
    
    if hp_values: # extend the title with HPs
        hp_suffix = (
            f" | seed={hp_values['seed']}, nf={hp_values['nf']}, "
            f"vt={hp_values['vt']}, nte={hp_values['nte']}"
        )
        title += hp_suffix

    combined_df = pd.concat(dfs, ignore_index=True)
    agg_df = combined_df.groupby("method")[metric].agg(["mean", "std"]).reset_index()

    agg_df["method_display"] = agg_df["method"].map(method_name_map)

    manual_methods = set(manual_df["method"].tolist())
    final_df = pd.concat([agg_df, manual_df], ignore_index=True)
    final_df_sorted = final_df.sort_values(by="mean")

    bar_colors = [manual_color if m in manual_methods else base_color for m in final_df_sorted["method"]]
    e_colors = [manual_e_color if m in manual_methods else base_e_color for m in final_df_sorted["method"]]

    fig_width = max(12, len(final_df_sorted) * 0.6)
    _, ax = plt.subplots(figsize=(fig_width, 6))
    x_pos = np.arange(len(final_df_sorted))

    ax.bar(
        x=x_pos,
        height=final_df_sorted["mean"],
        yerr=final_df_sorted["std"],
        color=bar_colors,
        capsize=0,
        ecolor=e_colors,
        linewidth=2
    )

    ax.grid(True, axis='y', linestyle='-', color='lightgray', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(final_df_sorted["method_display"], fontsize=11, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title().upper(), fontsize=12)
    ax.set_title(title, fontsize=13)

    y_min = (final_df_sorted["mean"] - final_df_sorted["std"]).min()
    y_max = (final_df_sorted["mean"] + final_df_sorted["std"]).max()
    y_range = y_max - y_min
    if y_range < 0.05:
        mid = (y_min + y_max) / 2
        y_min = mid - 0.025
        y_max = mid + 0.025

    tick_start = np.floor(y_min / 0.05) * 0.05
    tick_end = np.ceil(y_max / 0.05) * 0.05
    ticks = np.arange(tick_start, tick_end + 0.001, 0.05)
    if len(ticks) < 2:
        tick_end = tick_start + 0.05
        ticks = np.array([tick_start, tick_end])

    ax.set_ylim(ticks[0], ticks[-1])
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=1)
    for spine in ax.spines.values():
        spine.set_color('gray')
        spine.set_linewidth(1)

    legend_handles = [
        mpatches.Patch(color=base_color, label="TabPFN with sklearn methods"),
        mpatches.Patch(color=manual_color, label="TabArena models")
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        frameon=True,
        fontsize=11
    )
    legend.get_frame().set_edgecolor('gray')

    plt.tight_layout()
    plt.savefig(fig_abs_path)
    print(f"[INFO] Saved to {fig_abs_path}")

if __name__ == "__main__":
    fig_dir = "figures/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    method_name_map = {
        "original": "TabPFN",
        "random_fs": "TabPFN+randFS",
        "variance_fs": "TabPFN+varFS",
        "tree_fs": "TabPFN+treeFS",
        "pca_dr": "TabPFN+pca",
        "random_dr": "TabPFN+randproj",
        "agglo_dr": "TabPFN+agglo",
        "kbest_fs": "TabPFN+kbest",
        "ica_dr": "TabPFN+ica",
        "kpca_dr": "TabPFN+kpca"
    }
   
    datasets = [
        {
            "data_path": "./results_qsar/*.csv",
            "fig_path": f"QSAR_{timestamp}.pdf",
            "metric": "rmse",
            "title": "QSAR-TID-11(↓)",
            "manual_df": pd.DataFrame({
                "method": ["RealMLP", "TabM", "MNCA"],
                "method_display": ["RealMLP", "TabM", "MNCA"],
                "mean": [0.763, 0.761, 0.770],
                "std": [0.047, 0.050, 0.044]
            })
        },
        # {
        #     "data_path": "./results_bioresponse/*.csv",
        #     "fig_path": f"bioresponse_{timestamp}.pdf",
        #     "metric": "auc_roc",
        #     "title": "Bioresponse(↑)",
        #     "manual_df": pd.DataFrame({
        #         "method": ["RF", "XGBoost", "LightGBM", "CatBoost"],
        #         "method_display": ["RF", "XGBoost", "LightGBM", "CatBoost"],
        #         "mean": [0.873, 0.873, 0.872, 0.872],
        #         "std": [0.007, 0.008, 0.008, 0.009]
        #     })
        # },
        # {
        #     "data_path": "./results_hiva/*.csv",
        #     "fig_path": f"hiva_{timestamp}.pdf",
        #     "metric": "log_loss",
        #     "title": "hiva_agnostic(↓)",
        #     "manual_df": pd.DataFrame({
        #         "method": ["LightGBM", "EBM"],
        #         "method_display": ["LightGBM", "EBM"],
        #         "mean": [0.175, 0.174],
        #         "std": [0.001, 0.001]
        #     })        
        # }
    ]

    for dataset in datasets:
        plot_results(**dataset, fig_dir=fig_dir, method_name_map=method_name_map)