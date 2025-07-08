#!/usr/bin/env python3
"""
collect_training_times.py
-------------------------

Walks through <logs_dir>/ (recursively), parses every *.err file, extracts

  • Average-fold-time lines, e.g.
      Average fold time for 'random_fs': 58.0s over 4 folds
  • Total-elapsed lines, e.g.
      Total elapsed: 0h 3m 52.5s

and writes a CSV called 'training_time.csv' with columns

    log_file, method, avg_fold_time_s, total_elapsed_s

at <out_dir> (defaults to logs_dir).

Usage
-----
python collect_training_times.py \
    --logs_dir /work/dlclarge2/.../LOGS \
    --out_dir  /work/dlclarge2/.../analysis      # optional
"""
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- patterns --------------------------------------------------------
AVG_FOLD_RE   = re.compile(r"Average fold time for '([^']+)':\s*([0-9.]+)s")
TOTAL_ELAPSED = re.compile(
    r"Total elapsed:\s*(\d+)h\s*(\d+)m\s*([0-9.]+)s", re.I
)

# ---------- single-file parser ---------------------------------------------
def parse_err_file(path: Path) -> Optional[Tuple[str, float, float]]:
    """
    Return (method, avg_fold_sec, total_elapsed_sec) or None if not found.
    """
    method: Optional[str] = None
    avg_sec: Optional[float] = None
    total_sec: Optional[float] = None

    try:
        with path.open("r", errors="ignore") as f:
            for line in f:
                if avg_sec is None:
                    m = AVG_FOLD_RE.search(line)
                    if m:
                        method = m.group(1)
                        avg_sec = float(m.group(2))
                        continue
                if total_sec is None:
                    m = TOTAL_ELAPSED.search(line)
                    if m:
                        h, m_, s = m.groups()
                        total_sec = int(h) * 3600 + int(m_) * 60 + float(s)
                if avg_sec is not None and total_sec is not None:
                    break
    except Exception as e:  # pragma: no cover
        print(f"⚠️  Could not read {path}: {e}")

    if method and avg_sec is not None and total_sec is not None:
        return method, avg_sec, total_sec
    return None


# ---------- directory walker ------------------------------------------------
def gather_logs(logs_root: Path) -> List[Dict[str, str]]:
    """
    Recursively collect timing info from *.err files under logs_root.
    """
    records: List[Dict[str, str]] = []
    for err_file in logs_root.rglob("*.err"):
        parsed = parse_err_file(err_file)
        if parsed:
            method, avg_sec, total_sec = parsed
            records.append({
                "log_file": str(err_file.relative_to(logs_root)),
                "method": method,
                "avg_fold_time_s": f"{avg_sec:.3f}",
                "total_elapsed_s": f"{total_sec:.3f}",
            })
    return records


# ---------- CLI -------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Collect timing info from *.err logs.")
    ap.add_argument("--logs_dir", required=True,
                    help="Directory containing .err files (searched recursively).")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write training_time.csv (default: logs_dir).")
    args = ap.parse_args()

    logs_path = Path(args.logs_dir).expanduser().resolve()
    out_dir   = Path(args.out_dir or logs_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv   = out_dir / "training_time.csv"

    rows = gather_logs(logs_path)
    if not rows:
        print("No timing information found in *.err files.")
        return

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["log_file", "method",
                           "avg_fold_time_s", "total_elapsed_s"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
