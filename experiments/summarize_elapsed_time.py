#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

DISPLAY_NAME = {
    "tabpfnv2_tab": "TabPFNv2 TabArena",
    "tabpfnv2_org": "TabPFNv2 Original",
    "catboost_tab": "CatBoost TabArena",
}
def pretty(k: str) -> str: return DISPLAY_NAME.get(k, k)

# ── helpers ─────────────────────────────────────────────────────────────
def find_configs(dirs: list[Path]) -> list[Path]:
    cfgs = []
    for d in dirs:
        if d.exists():
            cfgs.extend(d.rglob("config.yaml"))
        else:
            logging.warning("%s does not exist – skipped.", d)
    logging.info("Found %d config.yaml files.", len(cfgs))
    return cfgs

NEEDED = ["dataset_name","method","model","avg_fold_time","total_elapsed_time"]
def parse_cfg(p: Path) -> dict[str, Any]:
    with p.open() as f: cfg = yaml.safe_load(f)
    row = {k: cfg.get(k) for k in NEEDED}
    if miss := [k for k,v in row.items() if v is None]:
        logging.warning("%s missing %s", p, miss)
    return row

def build_tables(files):
    df = pd.DataFrame([parse_cfg(p) for p in files])
    piv = {ds: (g.set_index(["method","model"])
                  [["avg_fold_time","total_elapsed_time"]]
                  .unstack("model").sort_index(axis=1).round(2))
           for ds,g in df.groupby("dataset_name")}
    return df, piv

# ── plotting ────────────────────────────────────────────────────────────
def plot_times(pivots, output_dir: Path, *, pdf=False, ref_model="TabPFNv2 TabArena"):
    import matplotlib.pyplot as plt, matplotlib as mpl
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = mpl.colormaps.get_cmap("tab10")
    ref_raw = next((r for r,n in DISPLAY_NAME.items() if ref_model in {r,n}), ref_model)

    for ds, pv in pivots.items():
        t = pv["total_elapsed_time"].copy()
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = t.columns.droplevel(0)
        t = t.rename(columns=pretty)

        models, methods = list(t.columns), list(t.index)
        fig, ax = plt.subplots(figsize=(max(6,1.4*len(models)+3),
                                        max(4,1+0.6*len(methods))))

        for i,meth in enumerate(methods):
            ys, xs = t.loc[meth], range(len(models))
            mask = ~ys.isna()
            ax.scatter([x for x,o in zip(xs,mask) if o], ys[mask],
                       s=60, color=cmap(i%cmap.N), edgecolor="#333",
                       label=meth, zorder=3)

        ref_pretty = pretty(ref_raw)
        if "original" in methods and ref_pretty in models \
           and pd.notna(t.loc["original", ref_pretty]):
            ax.axhline(t.loc["original", ref_pretty],
                       ls="--", lw=1.2, color="#555",
                       label="_nolegend_", zorder=2)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("Total elapsed time [s]")
        ax.set_title(ds); ax.grid(axis="y", ls="--", alpha=.4)
        h,l = ax.get_legend_handles_labels()
        ax.legend(h,l,frameon=False,ncol=max(1,len(l)//2),
                  bbox_to_anchor=(0.5,1.02),loc="lower center")
        plt.tight_layout()
        ext="pdf" if pdf else "png"
        plt.savefig(output_dir/f"{ds}.{ext}".lower(),
                    dpi=None if pdf else 300,bbox_inches="tight")
        plt.close()

# ── CLI ─────────────────────────────────────────────────────────────────
def main(argv: list[str]|None=None):
    pa=argparse.ArgumentParser(description="Summarise elapsed-time metrics.")
    pa.add_argument("--results_dir",nargs="+",required=True)
    pa.add_argument("--save_csv",action="store_true")
    pa.add_argument("--plot",action="store_true")
    pa.add_argument("--pdf",action="store_true")
    pa.add_argument("--ref_model",default="TabPFNv2 TabArena")
    pa.add_argument("--output_dir",default="experiments")
    a=pa.parse_args(argv)

    roots=[Path(d).expanduser().resolve() for d in a.results_dir]
    cfgs=find_configs(roots)
    if not cfgs:
        logging.error("No config.yaml files found — abort."); sys.exit(1)
    df,piv=build_tables(cfgs)

    base = Path(a.output_dir)/"time_summary"
    os.makedirs(base, exist_ok=True)
    figs = base/"figures"

    for ds,p in piv.items():
        print(f"\n=== {ds} ==="); print(p)
        if a.save_csv:
            p.to_csv(base/f"{ds.replace(' ','_')}.csv".lower())
            logging.info("Wrote %s", base/f"{ds}.csv".lower())

    if a.plot or a.pdf:
        plot_times(piv, figs, pdf=a.pdf, ref_model=a.ref_model)

if __name__=="__main__":
    main()
