#!/usr/bin/env python3
"""
prov_reuse_plotting.py - Demonstrate provenance-driven code reuse for plotting.

This example shows how the SAME plotting code can be reused across entirely
different experiments/use cases, because the PROV JSON provides a uniform
interface to experiment data regardless of the underlying ML task.

The provenance approach enables:
  1. A single plotting script that works for ANY experiment that produced PROV JSON
  2. Automatic discovery of metrics, parameters, and artifacts from provenance
  3. Comparison plots across different experiments (CIFAR vs MNIST vs custom)
  4. Sustainability/carbon reporting from any experiment
  5. RO-Crate unpacking: load a packaged experiment and plot its results

Usage:
    # Plot metrics from a single experiment's PROV data
    python examples/prov_reuse_plotting.py --prov-dir data/prov/my_experiment

    # Compare two experiments side by side
    python examples/prov_reuse_plotting.py \\
        --prov-dir data/prov/cifar_exp data/prov/mnist_exp \\
        --compare

    # Plot from an RO-Crate archive
    python examples/prov_reuse_plotting.py --ro-crate crates/run_0_ro-crate.zip

    # Generate all standard plots to a directory
    python examples/prov_reuse_plotting.py --prov-dir data/prov/my_experiment --out plots/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False


# ─────────────────────────────────────────────────────────────────────
# PROV DATA LOADING (uniform interface for any experiment)
# ─────────────────────────────────────────────────────────────────────

def load_prov_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_experiment_data(prov_dir: Path) -> List[Dict[str, Any]]:
    """Extract structured data from all PROV JSONs under a directory.

    Returns a list of run records, each with:
      - run_id, experiment_name
      - params: {name: value}
      - metrics: {name: value}  (from entity paths/labels)
      - source_file: path to the prov JSON
    """
    runs = []
    prov_files = sorted(glob.iglob(str(prov_dir / "**" / "prov_*.json"), recursive=True))

    for prov_path in prov_files:
        prov_path = Path(prov_path)
        try:
            doc = load_prov_json(prov_path)
        except Exception:
            continue

        run = {
            "source_file": str(prov_path),
            "experiment_name": "",
            "run_id": "",
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

        # Extract from activities
        for act_key, act_data in doc.get("activity", {}).items():
            if not isinstance(act_data, dict):
                continue

            for k, v in act_data.items():
                # Flatten typed values like {"$": "0.001", "type": "xsd:float"}
                if isinstance(v, dict) and "$" in v:
                    v = v["$"]

                if "experiment_name" in k:
                    run["experiment_name"] = str(v)
                elif "run_id" in k:
                    run["run_id"] = str(v)
                elif k.startswith("param:") or k.startswith("param_"):
                    pname = k.split(":", 1)[-1] if ":" in k else k.replace("param_", "")
                    run["params"][pname] = _try_numeric(v)
                elif not k.startswith("yProv4ML:") and not k.startswith("prov:"):
                    # Likely a hyperparameter
                    run["params"][k] = _try_numeric(v)

        # Extract metrics from entities
        for ent_key, ent_data in doc.get("entity", {}).items():
            if not isinstance(ent_data, dict):
                continue

            label = ent_data.get("yProv4ML:label", "")
            path_val = ent_data.get("yProv4ML:path", "")
            source = ent_data.get("yProv4ML:source", "")

            is_metric = (
                "METRIC" in str(source).upper()
                or "/metrics" in str(path_val).lower()
                or "metrics_" in str(path_val).lower()
            )

            if is_metric and label:
                # Try to read the metric file
                metric_file = _resolve_metric_file(prov_path, path_val)
                if metric_file:
                    val = _read_last_value(metric_file)
                    if val is not None:
                        run["metrics"][label] = val

            if "artifact" in str(path_val).lower():
                run["artifacts"].append(str(path_val))

        if run["experiment_name"] or run["params"] or run["metrics"]:
            runs.append(run)

    return runs


def _try_numeric(v):
    if isinstance(v, (int, float)):
        return v
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return float(v)
        except (ValueError, TypeError):
            return v


def _resolve_metric_file(prov_path: Path, metric_path: str) -> Optional[Path]:
    """Try to find the metric file relative to the PROV JSON."""
    if not metric_path:
        return None
    p = Path(metric_path)
    if p.is_absolute() and p.exists():
        return p

    base = prov_path.parent
    candidates = [
        base / metric_path,
        base / p.name,
    ]
    if metric_path.startswith("prov/"):
        rel = metric_path[5:]
        candidates.insert(0, base / rel)

    for c in candidates:
        if c.exists():
            return c
    return None


def _read_last_value(file_path: Path) -> Optional[float]:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        try:
            with open(file_path) as f:
                lines = f.readlines()
            if len(lines) < 2:
                return None
            for part in reversed(lines[-1].strip().split(",")):
                try:
                    return float(part.strip())
                except ValueError:
                    continue
        except Exception:
            pass
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
            pass
    return None


# ─────────────────────────────────────────────────────────────────────
# PLOTTING FUNCTIONS (work with ANY experiment data from PROV)
# ─────────────────────────────────────────────────────────────────────

def plot_metrics_bar(runs: List[Dict], out_dir: Optional[Path] = None, title: str = ""):
    """Bar chart of final metric values across runs."""
    if not MPL_OK or not runs:
        return

    # Collect all metric names
    all_metrics = set()
    for r in runs:
        all_metrics.update(r["metrics"].keys())

    if not all_metrics:
        print("  No metrics to plot")
        return

    # Filter to numeric metrics
    metric_names = sorted(all_metrics)
    n_metrics = len(metric_names)
    n_runs = len(runs)

    fig, axes = plt.subplots(1, min(n_metrics, 4), figsize=(4 * min(n_metrics, 4), 5))
    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metric_names[:4]):
        ax = axes[i]
        values = []
        labels = []
        for r in runs:
            if metric in r["metrics"]:
                values.append(r["metrics"][metric])
                label = r.get("run_id", "")[:8] or r.get("experiment_name", f"run_{len(labels)}")
                labels.append(label)

        if values:
            bars = ax.bar(range(len(values)), values, color=plt.cm.Set2(np.linspace(0, 1, len(values))))
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(metric, fontsize=10)
            ax.set_ylabel("Value")

    fig.suptitle(title or "Experiment Metrics Comparison", fontsize=12)
    plt.tight_layout()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "metrics_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_params_vs_metric(
    runs: List[Dict],
    metric_name: str = "accuracy",
    out_dir: Optional[Path] = None,
):
    """Scatter plots of each parameter vs a target metric."""
    if not MPL_OK or not runs:
        return

    # Find runs that have the target metric
    valid_runs = [r for r in runs if metric_name in r["metrics"]]
    if not valid_runs:
        # Try case-insensitive match
        for r in runs:
            for m in r["metrics"]:
                if metric_name.lower() in m.lower():
                    metric_name = m
                    break
            if metric_name in r["metrics"]:
                break
        valid_runs = [r for r in runs if metric_name in r["metrics"]]

    if not valid_runs:
        print(f"  No runs with metric '{metric_name}'")
        return

    # Find numeric parameters
    numeric_params = set()
    for r in valid_runs:
        for k, v in r["params"].items():
            if isinstance(v, (int, float)):
                numeric_params.add(k)

    if not numeric_params:
        print("  No numeric parameters to plot")
        return

    param_names = sorted(numeric_params)[:6]  # max 6 subplots
    n = len(param_names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]
        x_vals = []
        y_vals = []
        for r in valid_runs:
            if param in r["params"] and isinstance(r["params"][param], (int, float)):
                x_vals.append(r["params"][param])
                y_vals.append(r["metrics"][metric_name])

        if x_vals:
            ax.scatter(x_vals, y_vals, alpha=0.7, s=40)
            ax.set_xlabel(param)
            ax.set_ylabel(metric_name)
            ax.set_title(f"{param} vs {metric_name}", fontsize=10)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Parameter Impact on {metric_name}", fontsize=12)
    plt.tight_layout()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"params_vs_{metric_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_sustainability(runs: List[Dict], out_dir: Optional[Path] = None):
    """Plot sustainability metrics (emissions, energy) across runs."""
    if not MPL_OK or not runs:
        return

    sustainability_keys = ["emissions", "energy_consumed", "cpu_energy", "gpu_energy",
                           "ram_energy", "cpu_power", "gpu_power"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: emissions per run
    ax = axes[0]
    emissions = []
    labels = []
    for r in runs:
        for key in ["emissions", "emissions (tCO2eq)"]:
            if key in r["metrics"]:
                emissions.append(r["metrics"][key])
                labels.append(r.get("run_id", "")[:8] or f"run_{len(labels)}")
                break

    if emissions:
        ax.barh(range(len(emissions)), emissions, color="#4CAF50")
        ax.set_yticks(range(len(emissions)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Emissions (kg CO2eq)")
        ax.set_title("Carbon Emissions per Run")
    else:
        ax.text(0.5, 0.5, "No emissions data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Carbon Emissions per Run")

    # Right: energy breakdown (if available)
    ax = axes[1]
    energy_keys = ["cpu_energy", "gpu_energy", "ram_energy"]
    energy_data = defaultdict(list)
    run_labels = []
    for r in runs:
        has_energy = False
        for key in energy_keys:
            if key in r["metrics"]:
                energy_data[key].append(r["metrics"][key])
                has_energy = True
            else:
                energy_data[key].append(0)
        if has_energy:
            run_labels.append(r.get("run_id", "")[:8] or f"run_{len(run_labels)}")

    if run_labels:
        bottom = np.zeros(len(run_labels))
        colors = ["#2196F3", "#FF9800", "#9C27B0"]
        for i, key in enumerate(energy_keys):
            vals = energy_data[key][:len(run_labels)]
            ax.bar(range(len(run_labels)), vals, bottom=bottom, label=key, color=colors[i])
            bottom += np.array(vals)
        ax.set_xticks(range(len(run_labels)))
        ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Energy (kWh)")
        ax.set_title("Energy Breakdown per Run")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No energy data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Energy Breakdown per Run")

    plt.tight_layout()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "sustainability.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_experiment_comparison(
    experiments: Dict[str, List[Dict]],
    out_dir: Optional[Path] = None,
):
    """Compare metrics across different experiments (e.g., CIFAR vs MNIST)."""
    if not MPL_OK or not experiments:
        return

    # Collect all metrics across all experiments
    all_metrics = set()
    for runs in experiments.values():
        for r in runs:
            all_metrics.update(r["metrics"].keys())

    # Find metrics present in multiple experiments
    shared_metrics = []
    for m in sorted(all_metrics):
        count = sum(1 for runs in experiments.values()
                    if any(m in r["metrics"] for r in runs))
        if count > 1:
            shared_metrics.append(m)

    if not shared_metrics:
        print("  No shared metrics across experiments for comparison")
        return

    n = min(len(shared_metrics), 4)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    exp_names = list(experiments.keys())
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(exp_names)))

    for i, metric in enumerate(shared_metrics[:n]):
        ax = axes[i]
        for j, (exp_name, runs) in enumerate(experiments.items()):
            values = [r["metrics"][metric] for r in runs if metric in r["metrics"]]
            if values:
                positions = [j + k * 0.1 for k in range(len(values))]
                ax.scatter(positions, values, color=colors[j], label=exp_name, s=50, alpha=0.7)
                ax.axhline(np.mean(values), color=colors[j], linestyle="--", alpha=0.3)

        ax.set_title(metric, fontsize=10)
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)

    plt.suptitle("Cross-Experiment Comparison", fontsize=12)
    plt.tight_layout()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "experiment_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# RO-CRATE UNPACKING
# ─────────────────────────────────────────────────────────────────────

def load_from_ro_crate(crate_path: Path) -> Tuple[Path, List[Dict]]:
    """Unpack an RO-Crate and extract experiment data."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="rocrate_"))
    with zipfile.ZipFile(crate_path, "r") as zf:
        zf.extractall(tmp_dir)
    runs = extract_experiment_data(tmp_dir)
    return tmp_dir, runs


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Provenance-driven reusable plotting for any ML experiment."
    )
    parser.add_argument("--prov-dir", nargs="+", type=Path, default=None,
                        help="One or more provenance directories to plot")
    parser.add_argument("--ro-crate", type=Path, default=None,
                        help="RO-Crate ZIP file to unpack and plot")
    parser.add_argument("--compare", action="store_true",
                        help="Compare experiments when multiple --prov-dir given")
    parser.add_argument("--metric", type=str, default="accuracy",
                        help="Target metric for parameter analysis (default: accuracy)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for plots (displays interactively if omitted)")
    args = parser.parse_args()

    if not MPL_OK:
        print("matplotlib is required: pip install matplotlib")
        sys.exit(1)

    experiments = {}

    if args.ro_crate:
        print(f"Unpacking RO-Crate: {args.ro_crate}")
        tmp_dir, runs = load_from_ro_crate(args.ro_crate)
        experiments[args.ro_crate.stem] = runs
        print(f"  Found {len(runs)} runs")

    elif args.prov_dir:
        for pdir in args.prov_dir:
            print(f"Loading provenance: {pdir}")
            runs = extract_experiment_data(pdir)
            experiments[pdir.name] = runs
            print(f"  Found {len(runs)} runs, "
                  f"{sum(len(r['metrics']) for r in runs)} metric values")
    else:
        parser.error("Specify --prov-dir or --ro-crate")

    all_runs = [r for runs in experiments.values() for r in runs]

    if not all_runs:
        print("No experiment data found in provenance files.")
        return

    print(f"\nGenerating plots...")

    # Always generate these
    plot_metrics_bar(all_runs, out_dir=args.out, title="All Experiments")
    plot_params_vs_metric(all_runs, metric_name=args.metric, out_dir=args.out)
    plot_sustainability(all_runs, out_dir=args.out)

    # Cross-experiment comparison
    if args.compare and len(experiments) > 1:
        plot_experiment_comparison(experiments, out_dir=args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
