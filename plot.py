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


def plot_results(data_paths, fig_dir, fig_path, metric, manual_df, title, method_name_map, setting_labels):
    os.makedirs(fig_dir, exist_ok=True)
    fig_abs_path = os.path.join(fig_dir, fig_path)

    # Customize as needed
    setting_colors = ['#a6d8af', '#f4d8af', '#afc8f4']
    setting_edgecolors = ['#3e7949', '#a67b00', '#2b5db2']

    dfs = []
    hp_values = None

    for setting_idx, folder_pattern in enumerate(data_paths):
        setting_label = setting_labels[setting_idx]
        all_files = glob.glob(folder_pattern)
        if not all_files:
            print(f"[WARNING] No files matched: {folder_pattern}")
            continue

        for file in all_files:
            df = pd.read_csv(file)
            method_name = df["method"].iloc[0]
            df["method"] = method_name
            df["setting"] = setting_label
            dfs.append(df)

            if hp_values is None:
                hp_values = extract_hyperparams_from_filename(file)

    if not dfs:
        print("[ERROR] No data loaded.")
        return

    if hp_values:
        hp_suffix = (
            f" | seed={hp_values['seed']}, nf={hp_values['nf']}, "
            f"vt={hp_values['vt']}, nte={hp_values['nte']}"
        )
        # title += hp_suffix

    combined_df = pd.concat(dfs, ignore_index=True)
    agg_df = combined_df.groupby(["method", "setting"])[metric].agg(["mean", "std"]).reset_index()

    methods = sorted(agg_df["method"].unique())
    settings = setting_labels
    num_settings = len(settings)

    width = 0.2
    x = np.arange(len(methods))
    fig_width = max(12, len(methods) * 0.8)
    _, ax = plt.subplots(figsize=(fig_width, 6))

       # Plot method bars from CSVs (grouped per setting)
    for i, setting in enumerate(settings):
        df_s = agg_df[agg_df["setting"] == setting]
        df_s = df_s.set_index("method").reindex(methods).reset_index()

        ax.bar(
            x + (i - 1) * width,  # centered grouping
            df_s["mean"],
            width=width,
            yerr=df_s["std"],
            color=setting_colors[i],
            ecolor=setting_edgecolors[i],
            capsize=0,
            linewidth=2,
            label=setting
        )

    # === Plot manual_df as a separate group ===
    manual_methods = manual_df["method"].tolist()
    manual_x = np.arange(len(methods), len(methods) + len(manual_methods))
    ax.bar(
        manual_x,
        manual_df["mean"],
        width=width,
        yerr=manual_df["std"],
        color="#f4a6a6",
        ecolor="#c96a6a",
        capsize=0,
        linewidth=2,
        label="TabArena models"
    )

    # === Set combined x-axis labels ===
    method_labels = [method_name_map.get(m, m) for m in methods] + manual_df["method_display"].tolist()
    x_all = np.concatenate([x, manual_x])

    ax.set_xticks(x_all)
    ax.set_xticklabels(method_labels, fontsize=11, rotation=30, ha="right")

    ax.set_ylabel(metric.replace("_", " ").title().lower(), fontsize=12)
    ax.set_title(title, fontsize=13)

    all_means = pd.concat([agg_df["mean"], manual_df["mean"]])
    all_stds = pd.concat([agg_df["std"], manual_df["std"]])
    y_min = (all_means - all_stds).min()
    y_max = (all_means + all_stds).max()

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

    ax.grid(True, axis='y', linestyle='-', color='lightgray', linewidth=1)
    ax.set_axisbelow(True)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=1)
    for spine in ax.spines.values():
        spine.set_color('gray')
        spine.set_linewidth(1)

    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        frameon=True,
        fontsize=11
    ).get_frame().set_edgecolor('gray')

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
            "data_paths": [
               "./results_qsar/*.csv",
              "./results_qsar_150/*.csv",
              "./results_qsar_old/*.csv"
          ],
            "fig_path": f"QSAR_grouped_{timestamp}.pdf",
            "metric": "rmse",
            "title": "QSAR-TID-11(↓)",
            "manual_df": pd.DataFrame({
                "method": ["RealMLP", "TabM", "MNCA"],
                "method_display": ["RealMLP", "TabM", "MNCA"],
                "mean": [0.763, 0.761, 0.770],
                "std": [0.047, 0.050, 0.044]
            }),
            "setting_labels": ["sklearn new (500)", "sklearn new (150)", "sklearn old (500)"]
        },
        {
            "data_paths": [
                "./results_bioresponse/*.csv",
                "./results_bioresponse_150/*.csv",
                "./results_bioresponse_old/*.csv"
            ],
            "fig_path": f"bioresponse_grouped_{timestamp}.pdf",
            "metric": "auc_roc",
            "title": "Bioresponse (↑)",
            "manual_df": pd.DataFrame({
                "method": ["RF", "XGBoost", "LightGBM", "CatBoost"],
                "method_display": ["RF", "XGBoost", "LightGBM", "CatBoost"],
                "mean": [0.873, 0.873, 0.872, 0.872],
                "std": [0.007, 0.008, 0.008, 0.009]
            }),
            "setting_labels": ["sklearn new (500)", "sklearn new (150)", "sklearn old (500)"]
        },
        {
            "data_paths":  [
                "./results_hiva/*.csv",
                "./results_hiva_150/*.csv",
                "./results_hiva_old/*.csv"
            ],
            "fig_path": f"hiva_grouped_{timestamp}.pdf",
            "metric": "log_loss",
            "title": "hiva_agnostic(↓)",
            "manual_df": pd.DataFrame({
                "method": ["LightGBM", "EBM"],
                "method_display": ["LightGBM", "EBM"],
                "mean": [0.175, 0.174],
                "std": [0.001, 0.001]
            }),
            "setting_labels": ["sklearn new (500)", "sklearn new (150)", "sklearn old (500)"]
        }
    ]

    for dataset in datasets:
        plot_results(**dataset, fig_dir=fig_dir, method_name_map=method_name_map)