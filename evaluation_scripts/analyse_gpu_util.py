#!/usr/bin/env python3
"""
Analyze a single GPU utilization CSV log and write results to a file.

Expected columns (like your example):
    timestamp, util.gpu, util.mem, mem.used, mem.total

What it reports:
  - Basic stats (count, mean, std, min, max) for all numeric columns
  - Zero counts & fractions for util.gpu and util.mem
  - Average util.gpu / util.mem when values are non-zero
  - Memory usage percentage (mem.used / mem.total * 100) stats
  - Time span covered by the log (if timestamp exists)
  - Fraction of samples where GPU is active (util.gpu > 0)
  - Total number of samples

Usage:
    python analyze_gpu_csv.py path/to/file.csv

This will create:
    path/to/file_analysis.txt
"""

import argparse
from pathlib import Path

import pandas as pd
from datetime import datetime


def analyze_file(csv_path: Path) -> Path:
    # Output file in same directory, with _analysis suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = csv_path.with_name(f"{csv_path.stem}_analysis_{ts}.txt")

    # Read CSV, trimming spaces after commas (like in your example)
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    lines = []
    lines.append(f"Analysis for: {csv_path}")
    lines.append("=" * 80)
    lines.append(f"Total samples: {len(df)}")

    # Parse timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Ensure numeric columns are numeric
    for col in ["util.gpu", "util.mem", "mem.used", "mem.total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived column: memory usage percentage
    if "mem.used" in df.columns and "mem.total" in df.columns:
        df["mem.used_pct"] = df["mem.used"] / df["mem.total"] * 100.0

    # Basic stats for all numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append("")
        lines.append("Basic numeric stats (per column):")
        basic_stats = df[numeric_cols].agg(["count", "mean", "std", "min", "max"])
        lines.append(basic_stats.to_string(float_format=lambda x: f"{x:,.3f}"))
    else:
        lines.append("")
        lines.append("No numeric columns found – skipping basic stats.")

    # Detailed stats for util.gpu and util.mem
    util_details = {}
    for col in ["util.gpu", "util.mem"]:
        if col in df.columns:
            total_samples = df[col].shape[0]
            zeros = (df[col] == 0).sum()
            nonzero = df.loc[df[col] != 0, col]
            util_details[col] = {
                "zero_count": int(zeros),
                "zero_fraction": zeros / total_samples if total_samples else float("nan"),
                "nonzero_count": int(nonzero.count()),
                "nonzero_mean": nonzero.mean(),
                "max": df[col].max(),
            }

    if util_details:
        lines.append("")
        lines.append("util.gpu / util.mem details:")
        util_df = pd.DataFrame(util_details).T[
            ["zero_count", "zero_fraction", "nonzero_count", "nonzero_mean", "max"]
        ]
        lines.append(util_df.to_string(float_format=lambda x: f"{x:,.3f}"))

    # Memory usage percentage stats
    if "mem.used_pct" in df.columns:
        lines.append("")
        lines.append("Memory usage (% of total) stats:")
        mem_pct_stats = df["mem.used_pct"].agg(["count", "mean", "std", "min", "max"])
        lines.append(mem_pct_stats.to_string(float_format=lambda x: f"{x:,.3f}"))

    # Time span and sampling info
    if "timestamp" in df.columns:
        tmin = df["timestamp"].min()
        tmax = df["timestamp"].max()
        if pd.notna(tmin) and pd.notna(tmax):
            duration = tmax - tmin
            lines.append("")
            lines.append("Time span:")
            lines.append(f"  start : {tmin}")
            lines.append(f"  end   : {tmax}")
            lines.append(f"  range : {duration}")

    # Fraction of samples where GPU is "active"
    if "util.gpu" in df.columns:
        total = df["util.gpu"].shape[0]
        active = (df["util.gpu"] > 0).sum()
        if total:
            active_pct = active / total * 100.0
            lines.append("")
            lines.append(
                f"GPU active samples (util.gpu > 0): "
                f"{active}/{total} ({active_pct:.1f}%)"
            )

    # Write all lines to output file
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze a GPU utilization CSV log.")
    parser.add_argument(
        "csv_path",
        help="Path to a CSV file containing GPU utilization logs.",
    )
    args = parser.parse_args()

    csv_file = Path(args.csv_path)

    if not csv_file.is_file():
        print(f"Error: {csv_file} does not exist or is not a file.")
        raise SystemExit(1)

    output_path = analyze_file(csv_file)
    print(f"Analysis written to: {output_path}")


if __name__ == "__main__":
    main()