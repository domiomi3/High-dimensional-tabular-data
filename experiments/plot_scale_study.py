#!/usr/bin/env python3
"""
Scalability (row-level std) for TabPFNv2 with random feature selection.

- Pools ALL fold×repeat rows across runs per `num_features`
- Plots mean ± std vs number of features
- Std visuals (band/errorbars) use a lighter blue than the mean line.

Example:
  python plot_scalability_rows_std.py \
    --root /work/dlclarge2/matusd-toy_example/experiments/results/scalability_study/bioresponse/tabpfnv2_tab/random_fs \
    --out_png scalability_tabpfnv2_random_fs.png \
    --out_csv scalability_summary_rows.csv \
    --var_style both \
    --xtick_rotation 30
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import matplotlib.colors as mcolors

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # fallback parser will be used


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot ROC AUC vs #features with row-level std.")
    ap.add_argument(
        "--root",
        default="experiments/results/scalability_study/bioresponse/tabpfnv2_tab/random_fs",
        help="Directory containing *_results.csv and *_config.yaml files.",
    )
    ap.add_argument("--glob", default="*_results.csv", help="Glob for results CSV files.")
    ap.add_argument("--out_png", default="experiments/figures/scalability_study/bioresponse_tabpfnv2.png", help="Output PNG filename.")
    ap.add_argument("--out_csv", default="scalability_summary_rows.csv", help="Output CSV (row-level aggregates).")
    ap.add_argument("--title", default="TabPFNv2 with random feature selection", help="Plot title.")
    ap.add_argument(
        "--var_style",
        choices=["errorbar", "band", "both", "none"],
        default="errorbar",
        help="How to visualize variability (std across fold×repeat rows).",
    )
    ap.add_argument("--band_alpha", type=float, default=0.25, help="Alpha for shaded ±1 std band.")
    ap.add_argument("--errorbar_capsize", type=float, default=3.0, help="Caps for error bars.")
    ap.add_argument("--fig_w", type=float, default=6.0, help="Figure width (inches).")
    ap.add_argument("--fig_h", type=float, default=4.0, help="Figure height (inches).")
    ap.add_argument("--xtick_rotation", type=float, default=30.0, help="Rotate x tick labels (degrees).")
    ap.add_argument("--lighten_amount", type=float, default=0.6, help="0..1 mix with white for std color.")
    return ap.parse_args()


def read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    if yaml is not None:
        with path.open("r") as f:
            try:
                return yaml.safe_load(f)
            except Exception:
                pass
    # Very simple 'key: value' fallback
    data: Dict[str, Any] = {}
    try:
        txt = path.read_text()
        for m in re.finditer(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$", txt, flags=re.M):
            key = m.group(1)
            val = m.group(2)
            if re.fullmatch(r"-?\d+", val):
                data[key] = int(val)
            elif re.fullmatch(r"-?\d+\.\d+", val):
                data[key] = float(val)
            elif val.lower() in {"true", "false"}:
                data[key] = (val.lower() == "true")
            else:
                data[key] = val
        return data
    except Exception:
        return None


def find_in_mapping(mapping: Dict[str, Any], key: str) -> Optional[Any]:
    if not isinstance(mapping, dict):
        return None
    if key in mapping:
        return mapping[key]
    for v in mapping.values():
        if isinstance(v, dict):
            hit = find_in_mapping(v, key)
            if hit is not None:
                return hit
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    hit = find_in_mapping(item, key)
                    if hit is not None:
                        return hit
    return None


def infer_num_features_from_name(name: str) -> Optional[int]:
    m = re.search(r"num_features[_\- ]+(\d+)", name)
    return int(m.group(1)) if m else None


def lighten(color: str, amount: float = 0.6):
    """Lighten a matplotlib color by mixing with white (amount in [0,1])."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    out_png = Path(args.out_png)

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = root / out_csv

    csv_files = sorted(root.glob(args.glob))
    if not csv_files:
        print(f"❌ No files matched {args.glob} under {root}")
        return

    long_rows: List[pd.DataFrame] = []
    used = 0
    skipped = 0

    for csv_path in csv_files:
        stem = csv_path.name
        cfg_name = stem[:-len("_results.csv")] + "_config.yaml" if stem.endswith("_results.csv") else csv_path.stem + "_config.yaml"
        cfg_path = csv_path.with_name(cfg_name)

        # Determine num_features
        num_features: Optional[int] = None
        cfg = read_yaml(cfg_path)
        if isinstance(cfg, dict):
            val = find_in_mapping(cfg, "num_features")
            if isinstance(val, int):
                num_features = val
            elif isinstance(val, str) and val.isdigit():
                num_features = int(val)
        if num_features is None:
            num_features = infer_num_features_from_name(stem)
        if num_features is None:
            print(f"⚠️  Skip {csv_path.name}: cannot determine num_features (missing/invalid {cfg_path.name}).")
            skipped += 1
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠️  Skip {csv_path.name}: read error: {e}")
            skipped += 1
            continue

        if "roc_auc" not in df.columns:
            print(f"⚠️  Skip {csv_path.name}: 'roc_auc' column not found.")
            skipped += 1
            continue

        tmp = pd.DataFrame({
            "num_features": int(num_features),
            "roc_auc": pd.to_numeric(df["roc_auc"], errors="coerce"),
        }).dropna(subset=["roc_auc"])
        if len(tmp) == 0:
            print(f"⚠️  Skip {csv_path.name}: no valid roc_auc values.")
            skipped += 1
            continue

        long_rows.append(tmp)
        used += 1

    if not long_rows:
        print("❌ No valid rows found.")
        return

    rows_df = pd.concat(long_rows, ignore_index=True)

    # Row-level aggregation
    summary = (
        rows_df.groupby("num_features")["roc_auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("num_features")
        .rename(columns={"mean": "roc_auc_mean", "std": "roc_auc_std", "count": "n_rows"})
    )
    summary["roc_auc_std"] = summary["roc_auc_std"].fillna(0.0)
    summary.to_csv(out_csv, index=False)
    print(f"✓ Rows used: {used} file(s), skipped: {skipped} file(s)")
    print(f"✓ Wrote summary CSV: {out_csv}")

    # Colors: mean line in blue, std visuals in lighter blue
    base_color = "tab:blue"
    std_color = lighten(base_color, amount=args.lighten_amount)

    # Plot
    xs = summary["num_features"].values
    ys = summary["roc_auc_mean"].values
    ystd = summary["roc_auc_std"].values

    plt.figure(figsize=(args.fig_w, args.fig_h))
    line, = plt.plot(xs, ys, marker="o", label="Mean ROC AUC", color=base_color)

    if args.var_style in {"band", "both"}:
        plt.fill_between(xs, ys - ystd, ys + ystd, alpha=args.band_alpha, color=std_color, label="±1 std")

    if args.var_style in {"errorbar", "both"}:
        plt.errorbar(
            xs, ys, yerr=ystd,
            fmt="none",
            capsize=args.errorbar_capsize,
            elinewidth=1.0,
            ecolor=std_color,
        )
        if args.var_style == "errorbar":
            # Give the errorbar a legend entry if no band is drawn
            plt.plot([], [], color=std_color, label="±1 std")

    # Exact ticks at observed num_features and rotate
    ax = plt.gca()
    xticks = sorted(set(int(x) for x in xs))
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.set_xticklabels([str(x) for x in xticks])
    plt.setp(ax.get_xticklabels(), rotation=args.xtick_rotation, ha="right")

    plt.title(args.title)
    plt.xlabel("Number of features")
    plt.ylabel("ROC AUC")
    if args.var_style != "none":
        plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"✓ Wrote plot PNG: {out_png}")

    with pd.option_context("display.max_rows", None, "display.width", 140):
        print("\nRow-level summary by num_features:")
        print(summary)


if __name__ == "__main__":
    main()
