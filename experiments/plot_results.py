#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging, math, sys
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ------------------------------------------------------------------------ #
ROOT_DEFAULT = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIG_DIR      = Path(__file__).resolve().parents[1] / "experiments" / "figures"

MAXIMISE = {"roc_auc", "accuracy", "auc", "r2"}
MINIMISE = {"rmse", "mse", "mae", "log_loss", "norm_rmse"}

COL = {
    "TabPFNv2 Original": "#a6d8af",
    "TabPFNv2 TabArena": "#f4d8af",
    "CatBoost TabArena": "#afc8f4",
    "Baseline":          "#cd7ad0",
}
EDGE = "#333333"

TITLE_MAP = {
    "qsar_tid_11": "QSAR-TID-11",
    "bioresponse": "Bioresponse",
    "hiva_agnostic": "HIVA (Agnostic)",
}

# ------------------------------------------------------------------------ #
DISPLAY_NAME = {
    "tabpfnv2_tab": "TabPFNv2 TabArena",
    "tabpfnv2_org": "TabPFNv2 Original",
    "catboost_tab": "CatBoost TabArena",
}
def pretty(k: str) -> str: return DISPLAY_NAME.get(k, k)


def read_metric(csv: Path) -> tuple[str, float, float]:
    df = pd.read_csv(csv)
    metric = next(
        c for c in df.columns
        if c not in {
            "dataset_name", "model", "method",
            "repeat", "fold", "sample", "fold_time",
        }
    )
    return metric, df[metric].mean(), df[metric].std(ddof=0)


def abbrev_methods(meth: List[str]) -> str:
    if meth == ["all"]:
        return "all"
    if set(meth) == {"original"}:
        return "o"

    parts: list[str] = []
    if "original" in meth:
        parts.append("o")
    fs = sorted(m for m in meth if m.endswith("_fs"))
    dr = sorted(m for m in meth if m.endswith("_dr"))
    if fs:
        parts.append("fs" + "".join(x[0] for x in fs))
    if dr:
        parts.append("dr" + "".join(x[0] for x in dr))
    return "_".join(parts) or "custom"

# ------------------------------------------------------------------------ #
def discover_models(results_dir: Path) -> List[str]:
    return sorted(p.name for p in results_dir.iterdir() if p.is_dir())


def discover_methods(results_dir: Path, models: List[str]) -> List[str]:
    meth: set[str] = set()
    for m in models:
        meth.update(p.name for p in (results_dir / m).iterdir() if p.is_dir())
    return sorted(meth)

# ------------------------------------------------------------------------ #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",
                   help="Path like experiments/<group>/<dataset>")
    p.add_argument("--models",  nargs="+", default=["all"])
    p.add_argument("--methods", nargs="+", default=["all"])
    p.add_argument("--baseline_csv")
    p.add_argument("--ref_model", default="TabPFNv2 TabArena")
    p.add_argument("--pdf",  action="store_true",
                   help="Save as PDF (PNG otherwise)")
    a = p.parse_args()

    results_dir = Path(a.results_dir).resolve()
    if not results_dir.is_dir():
        sys.exit(f"❌ {results_dir} is not a directory")

    exp_group = results_dir.parent.name
    dataset   = results_dir.name

    # ─── models / methods discovery ──────────────────────────────────────
    models = (discover_models(results_dir)
              if a.models == ["all"] else a.models)
    if not models:
        sys.exit("❌ No model directories found.")

    methods = (discover_methods(results_dir, models)
               if a.methods == ["all"] else a.methods)

    if not models or not methods:
        sys.exit("❌ No models or methods found in the provided path.")

    # ─── baseline CSV -----------------------------------------------------
    if a.baseline_csv:
        baseline_csv = Path(a.baseline_csv)
    else:
        baseline_csv = results_dir / "sota.csv"

    has_base = baseline_csv.exists()

    # ─── load experiments -------------------------------------------------
    rows, metric_name = [], None
    for mdl in models:
        for meth in methods:
            csv = results_dir / mdl / meth / "results.csv"
            if csv.exists():
                m, mu, sd = read_metric(csv)
                metric_name = metric_name or m
                rows.append(dict(model=pretty(mdl), method=meth,
                                 mean=mu, std=sd))

    if not rows:
        sys.exit("❌ No experiment data found.")

    df = pd.DataFrame(rows)

    # ─── baseline ---------------------------------------------------------
    base_df = None
    if has_base:
        base_df = pd.read_csv(baseline_csv)
        if not {"model", "mean", "std"}.issubset(base_df.columns):
            sys.exit("❌ baseline_csv missing columns model,mean,std")
        base_df["colour"] = COL["Baseline"]

    # ─── ordering ---------------------------------------------------------
    minimise = metric_name in MINIMISE
    model_order = (
        df.groupby("model")["mean"].mean()
          .sort_values(ascending=minimise).index
    )
    ref_model = (
        a.ref_model if a.ref_model in model_order else model_order[0]
    )

    # robust to reference‐model missing some methods
    series = (df[df["model"] == ref_model]
                .set_index("method")["mean"]
                .loc[lambda s: s.index.isin(methods)])
    if series.empty:                     # fall back to global mean
        series = (df.groupby("method")["mean"].mean()
                    .loc[methods])
    method_order = series.sort_values(ascending=minimise).index.unique()

    df["model"]  = pd.Categorical(df["model"],  model_order,  ordered=True)
    df["method"] = pd.Categorical(df["method"], method_order, ordered=True)
    df.sort_values(["method", "model"], inplace=True)

    # ─── reference line value ───────────────────────────────────────────
    ref_line_val = None
    ref_orig = df[
        (df["model"] == ref_model) & (df["method"] == "original")
    ]
    if not ref_orig.empty:
        ref_line_val = ref_orig["mean"].iloc[0]

    # ─── bar positions ----------------------------------------------------
    n_base = len(base_df) if base_df is not None else 0
    n_meth, n_mod = len(method_order), len(model_order)

    bar_w  = 0.55 / n_mod
    x_base = np.arange(n_base) * (bar_w + 0.3)
    start_m = (x_base[-1] + bar_w if n_base else 0) + 0.6
    x_grp  = np.arange(n_meth) + start_m

    fig_width = max(5, 1.0 * (n_base + n_meth))   # ← slimmer figure
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    fig.subplots_adjust(top=0.99)
    if base_df is not None:
        for i, r in enumerate(base_df.itertuples()):
            ax.bar(
                x_base[i], r.mean, yerr=r.std, width=bar_w,
                color=COL["Baseline"], edgecolor=EDGE,
                error_kw=dict(ecolor=EDGE, capsize=2, lw=0.8),
            )

    for j, mdl in enumerate(model_order):
        sub = df[df["model"] == mdl]
        ax.bar(
            x_grp + j * bar_w, sub["mean"], yerr=sub["std"],
            width=bar_w, label=mdl,
            color=COL.get(mdl, "#aaaaaa"),
            edgecolor=EDGE,
            error_kw=dict(ecolor=EDGE, capsize=2, lw=0.8),
        )

    # ─── horizontal reference line (ref_model • original) ---------------
    if ref_line_val is not None:
        ax.axhline(
            ref_line_val,
            ls="--", lw=0.5, color="#555555",
        )

    centers = list(x_base) + list(x_grp + bar_w * (n_mod - 1) / 2)
    labels  = (
        list(base_df["model"]) if base_df is not None else []
    ) + list(method_order)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=30, ha="center")

    low  = min(df["mean"].min(),
               base_df["mean"].min() if base_df is not None else 1)
    high = max(
        (df["mean"] + df["std"]).max(),
        (base_df["mean"] + base_df["std"]).max() if base_df is not None else 0,
    )
    step  = 0.1
    y_min = math.floor(low  / step) * step
    y_max = math.ceil (high / step) * step
    ticks = np.arange(y_min, y_max + step/2, step)

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    arrow = "↓" if minimise else "↑" if metric_name in MAXIMISE else ""
    ax.set_ylabel(metric_name.replace("_", " "))
    ax.grid(axis="y", ls="--", alpha=0.3)

    # ─── legend: single horizontal row -----------------------------------
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        frameon=False,
        ncol=len(model_order),                 # single horizontal row
        bbox_to_anchor=(0.5, .98), loc="lower center",
    )

    nice_ds = TITLE_MAP.get(dataset, dataset.replace("_", " ").title())
    ax.set_title(f"{nice_ds} ({arrow})", pad=28)
    plt.tight_layout()

    # ─── save -------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # file name: <group>_<dataset>_<model1+model2>_<methodAbbr>.<ext>
    m_part = "+".join(models)
    if set(method_order) == set(methods):
        t_part = "all"
    else:
        t_part = abbrev_methods(list(method_order))

    ext    = "pdf" if a.pdf else "png"
    fname  = f"{exp_group}_{dataset}_{m_part}_{t_part}.{ext}"

    out_path = FIG_DIR / fname
    plt.savefig(out_path, dpi=300 if not a.pdf else None)
    print("✓ Figure saved to", out_path)


if __name__ == "__main__":
    main()
