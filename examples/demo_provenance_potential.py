#!/usr/bin/env python3
"""
demo_provenance_potential.py
============================
Self-contained demonstration of two key provenance advantages:

  1. SAME CODE, DIFFERENT USE CASE
     One plotting/analysis script that works identically on completely
     different ML tasks (image classification, text sentiment, time-series
     forecasting) because the PROV JSON provides a uniform data interface.

  2. RO-CRATE PACKAGING
     Snapshot the entire experiment (code + data + provenance) into a
     portable RO-Crate archive, then unpack it later to reproduce the
     exact plots -- even months later, on a different machine.

Run:
    python examples/demo_provenance_potential.py

Outputs everything under demo_output/.
No ML framework or GPU required -- uses synthetic PROV data.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import textwrap
import zipfile
from collections import OrderedDict
from pathlib import Path

# ── make src/ importable ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── optional deps ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    MPL_OK = True
except ImportError:
    MPL_OK = False


# ======================================================================
#  PART 0 — Generate synthetic PROV data for 3 different use cases
# ======================================================================

USE_CASES = {
    "image_classification": {
        "description": "CIFAR-10 CNN classification",
        "runs": [
            {"lr": 1e-3, "batch_size": 32, "epochs": 10, "accuracy": 0.91, "loss": 0.29,
             "emissions": 0.0038, "cpu_energy": 0.012, "gpu_energy": 0.006},
            {"lr": 1e-4, "batch_size": 64, "epochs": 10, "accuracy": 0.87, "loss": 0.41,
             "emissions": 0.0045, "cpu_energy": 0.014, "gpu_energy": 0.008},
            {"lr": 5e-4, "batch_size": 32, "epochs": 20, "accuracy": 0.93, "loss": 0.22,
             "emissions": 0.0072, "cpu_energy": 0.021, "gpu_energy": 0.014},
            {"lr": 1e-3, "batch_size": 128, "epochs": 5,  "accuracy": 0.85, "loss": 0.48,
             "emissions": 0.0021, "cpu_energy": 0.008, "gpu_energy": 0.003},
        ],
    },
    "text_sentiment": {
        "description": "IMDB sentiment analysis (BERT fine-tune)",
        "runs": [
            {"lr": 2e-5, "batch_size": 16, "epochs": 3, "accuracy": 0.89, "f1_score": 0.88,
             "emissions": 0.015, "cpu_energy": 0.032, "gpu_energy": 0.048},
            {"lr": 5e-5, "batch_size": 32, "epochs": 3, "accuracy": 0.91, "f1_score": 0.90,
             "emissions": 0.012, "cpu_energy": 0.028, "gpu_energy": 0.041},
            {"lr": 2e-5, "batch_size": 16, "epochs": 5, "accuracy": 0.92, "f1_score": 0.91,
             "emissions": 0.024, "cpu_energy": 0.051, "gpu_energy": 0.079},
        ],
    },
    "timeseries_forecast": {
        "description": "Energy demand LSTM forecasting",
        "runs": [
            {"lr": 1e-3, "hidden_size": 64,  "epochs": 50, "mae": 12.3, "rmse": 18.7,
             "emissions": 0.0051, "cpu_energy": 0.018, "gpu_energy": 0.0},
            {"lr": 5e-4, "hidden_size": 128, "epochs": 50, "mae": 10.1, "rmse": 15.2,
             "emissions": 0.0068, "cpu_energy": 0.022, "gpu_energy": 0.0},
            {"lr": 1e-3, "hidden_size": 128, "epochs": 100, "mae": 8.9,  "rmse": 13.5,
             "emissions": 0.0130, "cpu_energy": 0.041, "gpu_energy": 0.0},
        ],
    },
}

# Metric keys that are never hyperparameters
KNOWN_METRICS = {
    "accuracy", "loss", "f1_score", "mae", "rmse",
    "emissions", "cpu_energy", "gpu_energy", "ram_energy",
}


def _make_prov_json(experiment_name: str, run_idx: int, run_data: dict) -> dict:
    """Build a minimal W3C PROV JSON document from run data."""
    run_id = f"{experiment_name}_run_{run_idx}"

    params = {k: v for k, v in run_data.items() if k not in KNOWN_METRICS}
    metrics = {k: v for k, v in run_data.items() if k in KNOWN_METRICS}

    activity = OrderedDict()
    activity[f"yProv4ML:{run_id}"] = OrderedDict([
        ("yProv4ML:experiment_name", experiment_name),
        ("yProv4ML:run_id", run_id),
        ("yProv4ML:python_version", f"{sys.version_info.major}.{sys.version_info.minor}"),
    ])

    # Store params in a training context (mirrors real prov4ml output)
    training_ctx = OrderedDict()
    for k, v in params.items():
        training_ctx[k.upper()] = {"$": str(v), "type": "xsd:float"}
    activity["context:Context.TRAINING"] = training_ctx

    # Entities for each metric
    entities = OrderedDict()
    for mname, mval in metrics.items():
        eid = f"yProv4ML:{run_id}_{mname}"
        metric_path = f"prov/{run_id}/metrics/{mname}.csv"
        entities[eid] = OrderedDict([
            ("yProv4ML:label", mname),
            ("yProv4ML:path", metric_path),
            ("yProv4ML:source", "METRIC"),
        ])

    # Relations
    used = []
    wgb = []
    for eid in entities:
        wgb.append(OrderedDict([
            ("prov:entity", eid),
            ("prov:activity", f"yProv4ML:{run_id}"),
        ]))

    doc = OrderedDict([
        ("prefix", {"yProv4ML": "http://yprov4ml.2024/", "prov": "http://www.w3.org/ns/prov#"}),
        ("entity", entities),
        ("activity", activity),
        ("wasGeneratedBy", wgb),
    ])
    return doc, metrics


def generate_synthetic_data(base_dir: Path):
    """Write synthetic PROV JSONs + metric CSVs for all use cases."""
    for uc_name, uc_data in USE_CASES.items():
        for i, run in enumerate(uc_data["runs"]):
            run_dir = base_dir / uc_name / f"run_{i}"
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            # Write PROV JSON
            doc, metrics = _make_prov_json(uc_name, i, run)
            prov_path = run_dir / f"prov_{uc_name}_run_{i}.json"
            with open(prov_path, "w") as f:
                json.dump(doc, f, indent=2, default=str)

            # Write metric CSV files (so the plotting code can read them)
            for mname, mval in metrics.items():
                csv_path = metrics_dir / f"{mname}.csv"
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "value"])
                    w.writerow([0, mval])

    print(f"  Generated synthetic PROV data for {len(USE_CASES)} use cases")
    for uc_name, uc_data in USE_CASES.items():
        print(f"    {uc_name}: {len(uc_data['runs'])} runs — {uc_data['description']}")


# ======================================================================
#  PART 1 — SAME CODE, DIFFERENT USE CASE
# ======================================================================

def extract_runs_from_prov(prov_dir: Path) -> list:
    """Universal extraction: read any PROV JSON and return structured data."""
    import glob
    runs = []
    for prov_path in sorted(glob.iglob(str(prov_dir / "**" / "prov_*.json"), recursive=True)):
        prov_path = Path(prov_path)
        with open(prov_path) as f:
            doc = json.load(f)

        run = {"params": {}, "metrics": {}, "source": str(prov_path)}

        # Extract from activities
        for act_key, act_data in doc.get("activity", {}).items():
            if not isinstance(act_data, dict):
                continue
            for k, v in act_data.items():
                if isinstance(v, dict) and "$" in v:
                    v = v["$"]
                if "experiment_name" in k:
                    run["experiment"] = str(v)
                elif "run_id" in k:
                    run["run_id"] = str(v)

        # Extract params from training context
        training = doc.get("activity", {}).get("context:Context.TRAINING", {})
        for k, v in training.items():
            if isinstance(v, dict) and "$" in v:
                v = v["$"]
            try:
                run["params"][k.lower()] = float(v)
            except (ValueError, TypeError):
                run["params"][k.lower()] = v

        # Read metric values from CSV files
        for ent_key, ent_data in doc.get("entity", {}).items():
            if not isinstance(ent_data, dict):
                continue
            label = ent_data.get("yProv4ML:label", "")
            source = ent_data.get("yProv4ML:source", "")
            if "METRIC" not in str(source).upper():
                continue
            # Try to find the CSV file relative to the PROV JSON
            csv_path = prov_path.parent / "metrics" / f"{label}.csv"
            if csv_path.exists():
                try:
                    with open(csv_path) as cf:
                        lines = cf.readlines()
                    if len(lines) >= 2:
                        val = float(lines[-1].strip().split(",")[-1])
                        run["metrics"][label] = val
                except Exception:
                    pass

        if run["metrics"]:
            runs.append(run)
    return runs


def demo_same_code_different_usecase(data_dir: Path, out_dir: Path):
    """
    THE KEY DEMO: One function that produces meaningful plots for ANY experiment.
    No if/else for different tasks. The PROV interface is the abstraction.
    """
    print("\n" + "=" * 70)
    print("  DEMO 1: Same Plotting Code — Three Different ML Tasks")
    print("=" * 70)

    if not MPL_OK:
        print("  [matplotlib not installed — printing text summary instead]\n")
        for uc_name in USE_CASES:
            runs = extract_runs_from_prov(data_dir / uc_name)
            print(f"  --- {uc_name} ({len(runs)} runs) ---")
            for r in runs:
                params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
                metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in r["metrics"].items())
                print(f"    [{params_str}] => {metrics_str}")
        return

    # ── This is the SAME function applied to 3 completely different tasks ──

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, uc_name in enumerate(USE_CASES):
        runs = extract_runs_from_prov(data_dir / uc_name)
        desc = USE_CASES[uc_name]["description"]

        # ── Row 0: Metrics bar chart (works for any metrics) ──
        ax = axes[0][col]
        metric_names = sorted({m for r in runs for m in r["metrics"] if m not in KNOWN_METRICS - {"accuracy", "loss", "f1_score", "mae", "rmse"}})
        # Keep only the "result" metrics, not sustainability ones
        result_metrics = [m for m in metric_names if m not in {"emissions", "cpu_energy", "gpu_energy", "ram_energy"}]

        if result_metrics:
            x = np.arange(len(runs))
            width = 0.8 / len(result_metrics)
            colors = plt.cm.Set2(np.linspace(0, 0.8, len(result_metrics)))

            for j, metric in enumerate(result_metrics):
                vals = [r["metrics"].get(metric, 0) for r in runs]
                ax.bar(x + j * width, vals, width, label=metric, color=colors[j])

            ax.set_xticks(x + width * (len(result_metrics) - 1) / 2)
            ax.set_xticklabels([f"run_{i}" for i in range(len(runs))], fontsize=8)
            ax.legend(fontsize=7, loc="best")

        ax.set_title(f"{desc}", fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Metric Value", fontsize=9)

        # ── Row 1: Emissions vs performance (works for any task) ──
        ax = axes[1][col]
        perf_metric = None
        for candidate in ["accuracy", "f1_score", "mae", "rmse"]:
            if any(candidate in r["metrics"] for r in runs):
                perf_metric = candidate
                break

        if perf_metric:
            perf_vals = [r["metrics"].get(perf_metric, 0) for r in runs]
            emit_vals = [r["metrics"].get("emissions", 0) for r in runs]

            scatter = ax.scatter(emit_vals, perf_vals, s=120, c=range(len(runs)),
                                 cmap="viridis", edgecolors="black", linewidths=0.5, zorder=5)
            for k, r in enumerate(runs):
                ax.annotate(f"run_{k}", (emit_vals[k], perf_vals[k]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)

            ax.set_xlabel("Emissions (kg CO2eq)", fontsize=9)
            ax.set_ylabel(perf_metric, fontsize=9)
            ax.set_title(f"Sustainability vs {perf_metric}", fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Same Plotting Code Applied to 3 Different ML Tasks\n"
        "(provenance provides a uniform data interface)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = out_dir / "demo1_same_code_different_usecases.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Plot saved: {out_path}")
    print("  Key insight: ZERO task-specific code in the plotting function.")
    print("  The PROV JSON is the universal interface between experiment and analysis.")


# ======================================================================
#  PART 2 — RO-CRATE PACKAGING + REPLAY
# ======================================================================

def _build_ro_crate_metadata(run_name, files, git_commit="demo"):
    """Minimal RO-Crate 1.1 metadata."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    graph = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "about": {"@id": "./"},
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
        },
        {
            "@id": "./",
            "@type": "Dataset",
            "name": f"RO-Crate: {run_name}",
            "description": f"Provenance snapshot for {run_name}",
            "datePublished": now,
            "hasPart": [{"@id": f["path"]} for f in files],
        },
    ]
    for f in files:
        graph.append({
            "@id": f["path"],
            "@type": "File",
            "name": os.path.basename(f["path"]),
            "description": f.get("description", ""),
        })
    return {"@context": "https://w3id.org/ro/crate/1.1/context", "@graph": graph}


def demo_ro_crate_packaging(data_dir: Path, out_dir: Path):
    """
    THE KEY DEMO: Package an experiment into an RO-Crate, then unpack it
    on a "fresh" machine and reproduce the exact same plots.
    """
    print("\n" + "=" * 70)
    print("  DEMO 2: RO-Crate Packaging — Snapshot & Replay")
    print("=" * 70)

    # ── Step 1: Pick one experiment and package it ──
    uc_name = "image_classification"
    uc_dir = data_dir / uc_name
    crate_dir = out_dir / "crates"
    crate_dir.mkdir(parents=True, exist_ok=True)

    # Collect all files
    files = []
    for f in sorted(uc_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(uc_dir)
            files.append({
                "path": str(rel).replace("\\", "/"),
                "abs_path": str(f),
                "description": "PROV data" if f.suffix == ".json" else "Metric file",
            })

    # Also include the plotting script itself (the "code snapshot")
    plotting_script = ROOT / "examples" / "demo_provenance_potential.py"
    if plotting_script.exists():
        files.append({
            "path": "src/demo_provenance_potential.py",
            "abs_path": str(plotting_script),
            "description": "Analysis/plotting script (code snapshot)",
        })

    # Build and write RO-Crate
    metadata = _build_ro_crate_metadata(uc_name, files)
    crate_path = crate_dir / f"{uc_name}_snapshot.zip"

    with zipfile.ZipFile(crate_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ro-crate-metadata.json", json.dumps(metadata, indent=2))
        for f in files:
            zf.write(f["abs_path"], f["path"])

    crate_size = crate_path.stat().st_size
    print(f"\n  Step 1: Created RO-Crate archive")
    print(f"    Path:  {crate_path}")
    print(f"    Size:  {crate_size / 1024:.1f} KB")
    print(f"    Files: {len(files)} ({len(files)-1} data + 1 code snapshot)")

    # ── Step 2: Simulate "fresh machine" — unpack to a temp dir ──
    replay_dir = out_dir / "replay_from_crate"
    if replay_dir.exists():
        shutil.rmtree(replay_dir)
    replay_dir.mkdir(parents=True)

    with zipfile.ZipFile(crate_path, "r") as zf:
        zf.extractall(replay_dir)

    # Read and display the metadata
    with open(replay_dir / "ro-crate-metadata.json") as f:
        crate_meta = json.load(f)

    print(f"\n  Step 2: Unpacked RO-Crate to simulate fresh environment")
    print(f"    Replay dir: {replay_dir}")
    print(f"    RO-Crate conformsTo: {crate_meta['@graph'][0]['conformsTo']['@id']}")
    print(f"    Contents:")
    for item in crate_meta["@graph"][2:]:
        print(f"      {item['@id']}  — {item.get('description', '')}")

    # ── Step 3: Re-run the analysis from the unpacked crate ──
    print(f"\n  Step 3: Reproducing analysis from unpacked crate...")

    # The key point: use the SAME extract function on unpacked data
    replayed_runs = extract_runs_from_prov(replay_dir)
    print(f"    Extracted {len(replayed_runs)} runs from unpacked crate")

    if MPL_OK and replayed_runs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: metrics from crate
        metric_names = sorted({m for r in replayed_runs for m in r["metrics"]
                               if m not in {"emissions", "cpu_energy", "gpu_energy"}})
        x = np.arange(len(replayed_runs))
        width = 0.8 / max(len(metric_names), 1)
        colors = plt.cm.Set2(np.linspace(0, 0.8, max(len(metric_names), 1)))

        for j, metric in enumerate(metric_names):
            vals = [r["metrics"].get(metric, 0) for r in replayed_runs]
            ax1.bar(x + j * width, vals, width, label=metric, color=colors[j])
        ax1.set_xticks(x + width * (len(metric_names) - 1) / 2)
        ax1.set_xticklabels([f"run_{i}" for i in range(len(replayed_runs))], fontsize=9)
        ax1.set_title("Metrics (reproduced from RO-Crate)", fontsize=11)
        ax1.legend(fontsize=8)
        ax1.set_ylabel("Value")

        # Plot 2: sustainability from crate
        emit = [r["metrics"].get("emissions", 0) for r in replayed_runs]
        acc = [r["metrics"].get("accuracy", 0) for r in replayed_runs]
        sc = ax2.scatter(emit, acc, s=150, c=range(len(replayed_runs)),
                         cmap="RdYlGn", edgecolors="black", linewidths=1, zorder=5)
        for k in range(len(replayed_runs)):
            ax2.annotate(f"run_{k}", (emit[k], acc[k]),
                         textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax2.set_xlabel("Emissions (kg CO2eq)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs Emissions (from RO-Crate)", fontsize=11)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            "Plots Reproduced From RO-Crate Archive\n"
            "(unpacked on a 'fresh machine' — exact same results)",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()

        out_path = out_dir / "demo2_ro_crate_replay.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Plot saved: {out_path}")
    else:
        print("    [matplotlib not installed — text summary]")
        for r in replayed_runs:
            metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in r["metrics"].items())
            print(f"      {metrics_str}")

    print("\n  Key insight: The RO-Crate bundles code + data + provenance.")
    print("  Months later, on any machine, unzip and reproduce the exact analysis.")


# ======================================================================
#  MAIN
# ======================================================================

def main():
    out_dir = ROOT / "demo_output"
    data_dir = out_dir / "synthetic_prov"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    print("=" * 70)
    print("  yProv4ML — Demonstrating the Potential of Provenance")
    print("=" * 70)

    # Generate synthetic data
    print("\nGenerating synthetic PROV data for 3 ML use cases...")
    generate_synthetic_data(data_dir)

    # Demo 1: same code, different use case
    demo_same_code_different_usecase(data_dir, out_dir)

    # Demo 2: RO-Crate packaging
    demo_ro_crate_packaging(data_dir, out_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(textwrap.dedent("""
    Provenance (W3C PROV) provides a UNIFORM INTERFACE to experiment data:

    1. SAME CODE, DIFFERENT USE CASE
       - Image classification, text sentiment, time-series forecasting
       - The plotting function has ZERO task-specific logic
       - PROV JSON abstracts away the differences between experiments
       - Any new experiment that produces PROV data works automatically

    2. RO-CRATE PACKAGING
       - Snapshot code + PROV + metrics into a portable ZIP archive
       - Months later: unzip, run the same analysis, get identical results
       - Perfect for paper submissions, reproducibility, collaboration
       - The code snapshot IS the analysis tool (no version mismatch)
    """))

    print(f"  All outputs: {out_dir}/")
    for f in sorted(out_dir.rglob("*.png")):
        print(f"    {f.relative_to(out_dir)}")
    for f in sorted(out_dir.rglob("*.zip")):
        print(f"    {f.relative_to(out_dir)}")
    print()


if __name__ == "__main__":
    main()
