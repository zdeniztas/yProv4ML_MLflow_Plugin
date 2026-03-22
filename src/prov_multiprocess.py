#!/usr/bin/env python3
"""
prov_multiprocess.py - Handle multi-process provenance files.

When training runs use multiple processes (e.g., distributed data parallel),
prov4ml generates separate metrics directories per process rank:
  metrics_GR0/, metrics_GR1/, metrics_GR2/, ...

This utility:
  1. Discovers all process-specific provenance data under a run directory
  2. Joins them into a unified provenance document without duplicating shared
     entities (model, dataset, config) while keeping per-process metrics separate
  3. Generates post-processing statistics (aggregated across processes)
  4. Outputs a unified JSON + optional CSV summary

Usage:
    # Join multi-process provenance for a single run
    python src/prov_multiprocess.py --run-dir data/prov/experiment/run_0 --out merged/

    # Join all runs under an experiment with post-processing stats
    python src/prov_multiprocess.py --experiment-dir data/prov/my_experiment --out merged/ --stats

    # Full workflow: join + statistics + CSV export
    python src/prov_multiprocess.py --experiment-dir data/prov/my_experiment --out merged/ --stats --csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

# Reuse joining logic from prov_join
try:
    from prov_join import join_prov_documents, load_prov_json, find_prov_jsons
except ImportError:
    # Fallback: inline minimal versions
    def load_prov_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def find_prov_jsons(root):
        return sorted(Path(p) for p in glob.iglob(str(root / "**" / "prov_*.json"), recursive=True))


def discover_process_dirs(run_dir: Path) -> Dict[str, Path]:
    """Find all metrics_GR* directories in a run directory.

    Returns dict mapping process tag (e.g., "GR0") to its metrics directory.
    """
    process_dirs = {}
    for entry in sorted(run_dir.iterdir()):
        if entry.is_dir():
            m = re.match(r'^metrics_(GR\d+)$', entry.name)
            if m:
                process_dirs[m.group(1)] = entry
    # Also check for a plain "metrics" directory (single-process case)
    plain = run_dir / "metrics"
    if plain.is_dir() and not process_dirs:
        process_dirs["GR0"] = plain
    return process_dirs


def read_metric_value(file_path: Path) -> Optional[float]:
    """Read the last value from a metric file (CSV or NetCDF)."""
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            if len(lines) < 2:
                return None
            # Try to parse last line, last numeric column
            parts = lines[-1].strip().split(",")
            for p in reversed(parts):
                try:
                    return float(p.strip())
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    elif suffix == ".nc":
        try:
            import xarray as xr
            ds = xr.open_dataset(file_path)
            var = "values" if "values" in ds.data_vars else list(ds.data_vars)[0]
            vals = ds[var].values.flatten()
            ds.close()
            valid = vals[~np.isnan(vals)]
            return float(valid[-1]) if len(valid) > 0 else None
        except Exception:
            return None

    return None


def collect_process_metrics(process_dirs: Dict[str, Path]) -> Dict[str, Dict[str, float]]:
    """Collect all metric values from each process directory.

    Returns: {process_tag: {metric_name: value}}
    """
    all_metrics = {}
    for tag, mdir in process_dirs.items():
        metrics = {}
        for f in sorted(mdir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".csv", ".nc"):
                # Extract metric name from filename
                name = f.stem
                # Strip context suffix if present (e.g., "loss_Context.TRAINING" -> "loss")
                if "_Context" in name:
                    name = name.split("_Context")[0]
                val = read_metric_value(f)
                if val is not None:
                    metrics[name] = val
        all_metrics[tag] = metrics
    return all_metrics


def compute_aggregate_stats(
    process_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Compute aggregate statistics across processes for each metric.

    Returns: {metric_name: {mean, std, min, max, sum, count}}
    """
    # Collect all values per metric
    metric_values: Dict[str, List[float]] = defaultdict(list)
    for tag, metrics in process_metrics.items():
        for name, val in metrics.items():
            metric_values[name].append(val)

    stats = {}
    for name, values in sorted(metric_values.items()):
        arr = np.array(values)
        stats[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sum": float(np.sum(arr)),
            "count": len(values),
        }
    return stats


def join_run_provenance(
    run_dir: Path,
    out_dir: Path,
    include_stats: bool = False,
) -> Optional[dict]:
    """Join multi-process provenance for a single run.

    Returns the merged document (or None if no PROV files found).
    """
    # Find prov JSON files
    prov_files = find_prov_jsons(run_dir)
    if not prov_files:
        return None

    # Discover process directories
    process_dirs = discover_process_dirs(run_dir)

    # Load and join PROV documents
    docs = []
    paths = []
    for p in prov_files:
        try:
            docs.append(load_prov_json(p))
            paths.append(str(p))
        except Exception:
            pass

    if not docs:
        return None

    # Use the join utility with multi-process mode
    merged = join_prov_documents(
        docs,
        source_paths=paths,
        tag_sources=True,
        multi_process=len(process_dirs) > 1,
    )

    # Collect per-process metrics and add to the merged document
    if process_dirs:
        process_metrics = collect_process_metrics(process_dirs)
        merged["_process_metrics"] = process_metrics

        if include_stats:
            stats = compute_aggregate_stats(process_metrics)
            merged["_aggregate_stats"] = stats

    return merged


def process_experiment(
    experiment_dir: Path,
    out_dir: Path,
    include_stats: bool = False,
    export_csv: bool = False,
):
    """Process all runs under an experiment directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        d for d in experiment_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not run_dirs:
        print(f"No run directories found under {experiment_dir}")
        return

    all_run_stats = []

    for run_dir in run_dirs:
        print(f"\nProcessing run: {run_dir.name}")
        merged = join_run_provenance(run_dir, out_dir, include_stats=include_stats)

        if merged is None:
            print(f"  No PROV data found, skipping")
            continue

        # Write merged JSON
        out_path = out_dir / f"{run_dir.name}_merged.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"  Merged PROV -> {out_path}")

        # Collect stats for CSV
        if include_stats and "_aggregate_stats" in merged:
            row = {"run": run_dir.name}
            for metric, stat_vals in merged["_aggregate_stats"].items():
                row[f"{metric}_mean"] = stat_vals["mean"]
                row[f"{metric}_std"] = stat_vals["std"]
                row[f"{metric}_min"] = stat_vals["min"]
                row[f"{metric}_max"] = stat_vals["max"]
            all_run_stats.append(row)

            print(f"  Aggregate stats:")
            for metric, sv in merged["_aggregate_stats"].items():
                print(f"    {metric}: mean={sv['mean']:.6g}, std={sv['std']:.6g}, "
                      f"[{sv['min']:.6g}, {sv['max']:.6g}] (n={sv['count']})")

    # Export CSV summary
    if export_csv and all_run_stats and PANDAS_OK:
        csv_path = out_dir / "experiment_summary.csv"
        df = pd.DataFrame(all_run_stats)
        df.to_csv(csv_path, index=False)
        print(f"\nExperiment summary CSV -> {csv_path}")
    elif export_csv and not PANDAS_OK:
        print("\npandas not installed, skipping CSV export")


def main():
    parser = argparse.ArgumentParser(
        description="Handle multi-process provenance files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path,
                       help="Single run directory with multi-process metrics")
    group.add_argument("--experiment-dir", type=Path,
                       help="Experiment directory containing multiple run directories")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for merged files")
    parser.add_argument("--stats", action="store_true",
                        help="Compute aggregate statistics across processes")
    parser.add_argument("--csv", action="store_true",
                        help="Export summary CSV (requires pandas)")
    args = parser.parse_args()

    if args.run_dir:
        merged = join_run_provenance(args.run_dir, args.out, include_stats=args.stats)
        if merged:
            args.out.mkdir(parents=True, exist_ok=True)
            out_path = args.out / f"{args.run_dir.name}_merged.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, default=str)
            print(f"Merged PROV -> {out_path}")
        else:
            print("No PROV data found")
    else:
        process_experiment(args.experiment_dir, args.out,
                           include_stats=args.stats, export_csv=args.csv)


if __name__ == "__main__":
    main()
