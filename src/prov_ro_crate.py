#!/usr/bin/env python3
"""
prov_ro_crate.py - Package experiment code and provenance into RO-Crate archives.

An RO-Crate (Research Object Crate) is a lightweight approach to packaging
research data with their metadata. This is useful for:
  - Creating reproducible snapshots of the codebase at experiment time
  - Sharing experiments with full provenance context
  - Archiving training scripts + model + metrics + PROV documents
  - Reusing the exact code that produced specific results (e.g., for plotting)

This tool creates RO-Crate-compatible ZIP archives containing:
  - The training script(s) used
  - PROV JSON documents
  - Metric files (CSV/NetCDF)
  - Model artifacts
  - An ro-crate-metadata.json descriptor

Usage:
    # Package a single run
    python src/prov_ro_crate.py --run-dir data/prov/experiment/run_0 --out crates/

    # Package with additional source files
    python src/prov_ro_crate.py --run-dir data/prov/experiment/run_0 \\
        --include-src examples/demo.py yprov_mlflow_plugin/ \\
        --out crates/

    # Package all runs in an experiment
    python src/prov_ro_crate.py --experiment-dir data/prov/my_experiment --out crates/

    # Package with git snapshot (captures current commit info)
    python src/prov_ro_crate.py --run-dir data/prov/experiment/run_0 --out crates/ --git-info
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_git_info() -> Optional[Dict[str, str]]:
    """Get current git commit info if in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        remote = ""
        try:
            remote = subprocess.check_output(
                ["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            pass
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(dirty),
            "remote": remote,
        }
    except Exception:
        return None


def _build_ro_crate_metadata(
    run_name: str,
    files: List[Dict[str, Any]],
    description: str = "",
    git_info: Optional[Dict[str, str]] = None,
) -> dict:
    """Build an ro-crate-metadata.json conforming to RO-Crate 1.1 spec."""
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
            "description": description or f"Provenance-packaged experiment run: {run_name}",
            "datePublished": now,
            "creator": [{"@id": "#yProv4ML"}],
            "hasPart": [{"@id": f["path"]} for f in files],
        },
        {
            "@id": "#yProv4ML",
            "@type": "SoftwareApplication",
            "name": "yProv4ML MLflow Plugin",
            "url": "https://github.com/HPCI-Lab/yProv4ML_MLflow_Plugin",
        },
    ]

    # Add file entries
    for f in files:
        entry = {
            "@id": f["path"],
            "@type": "File",
            "name": os.path.basename(f["path"]),
        }
        if "description" in f:
            entry["description"] = f["description"]
        if "encodingFormat" in f:
            entry["encodingFormat"] = f["encodingFormat"]
        if "contentSize" in f:
            entry["contentSize"] = f["contentSize"]
        graph.append(entry)

    # Add git info as contextual entity
    if git_info:
        graph.append({
            "@id": "#git-snapshot",
            "@type": "SoftwareSourceCode",
            "name": "Git Snapshot",
            "version": git_info.get("commit", "unknown"),
            "codeRepository": git_info.get("remote", ""),
            "description": (
                f"Branch: {git_info.get('branch', 'unknown')}, "
                f"Dirty: {git_info.get('dirty', False)}"
            ),
        })
        graph[1]["isBasedOn"] = {"@id": "#git-snapshot"}

    return {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": graph,
    }


def _guess_encoding_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    formats = {
        ".json": "application/json",
        ".csv": "text/csv",
        ".nc": "application/x-netcdf",
        ".py": "text/x-python",
        ".pt": "application/octet-stream",
        ".pkl": "application/octet-stream",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".toml": "text/toml",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".dot": "text/vnd.graphviz",
    }
    return formats.get(ext, "application/octet-stream")


def _collect_run_files(run_dir: Path) -> List[Dict[str, Any]]:
    """Collect all relevant files from a run directory."""
    files = []
    for f in sorted(run_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            rel = f.relative_to(run_dir)
            files.append({
                "path": str(rel).replace("\\", "/"),
                "abs_path": str(f),
                "encodingFormat": _guess_encoding_format(str(f)),
                "contentSize": f.stat().st_size,
                "description": _describe_file(rel),
            })
    return files


def _describe_file(rel_path: Path) -> str:
    """Generate a human-readable description for a file based on its path."""
    name = rel_path.name
    parts = rel_path.parts

    if name.startswith("prov_") and name.endswith(".json"):
        return "W3C PROV provenance document"
    if "metrics" in str(rel_path).lower():
        return f"Metric data: {rel_path.stem}"
    if "artifact" in str(rel_path).lower():
        return f"Experiment artifact: {name}"
    if name.endswith(".pt") or name.endswith(".pkl"):
        return "Trained model checkpoint"
    if name == "config.json":
        return "Experiment configuration"
    return f"Experiment file: {name}"


def create_ro_crate(
    run_dir: Path,
    out_dir: Path,
    include_src: Optional[List[Path]] = None,
    git_info: Optional[Dict[str, str]] = None,
    run_name: Optional[str] = None,
) -> Path:
    """Create an RO-Crate ZIP archive for a run directory.

    Returns the path to the created ZIP file.
    """
    run_name = run_name or run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect run files
    files = _collect_run_files(run_dir)

    # Add extra source files
    src_files = []
    if include_src:
        for src_path in include_src:
            src_path = Path(src_path)
            if src_path.is_file():
                src_files.append({
                    "path": f"src/{src_path.name}",
                    "abs_path": str(src_path.resolve()),
                    "encodingFormat": _guess_encoding_format(str(src_path)),
                    "contentSize": src_path.stat().st_size,
                    "description": f"Source code: {src_path.name}",
                })
            elif src_path.is_dir():
                for f in sorted(src_path.rglob("*.py")):
                    rel = f.relative_to(src_path.parent)
                    src_files.append({
                        "path": f"src/{str(rel).replace(os.sep, '/')}",
                        "abs_path": str(f.resolve()),
                        "encodingFormat": "text/x-python",
                        "contentSize": f.stat().st_size,
                        "description": f"Source code: {rel}",
                    })

    all_files = files + src_files

    # Build metadata
    metadata = _build_ro_crate_metadata(
        run_name=run_name,
        files=all_files,
        git_info=git_info,
    )

    # Create ZIP
    zip_name = f"{run_name}_ro-crate.zip"
    zip_path = out_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add metadata
        zf.writestr("ro-crate-metadata.json", json.dumps(metadata, indent=2))

        # Add data files
        for f in all_files:
            zf.write(f["abs_path"], f["path"])

    print(f"  RO-Crate: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Files: {len(all_files)} ({len(files)} run + {len(src_files)} source)")
    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Package experiment provenance into RO-Crate archives."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path,
                       help="Single run directory to package")
    group.add_argument("--experiment-dir", type=Path,
                       help="Experiment directory containing multiple runs")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for RO-Crate ZIP files")
    parser.add_argument("--include-src", nargs="+", type=Path, default=None,
                        help="Additional source files/directories to include")
    parser.add_argument("--git-info", action="store_true",
                        help="Include current git commit information")
    args = parser.parse_args()

    git = _get_git_info() if args.git_info else None

    if args.run_dir:
        print(f"Packaging run: {args.run_dir.name}")
        create_ro_crate(args.run_dir, args.out,
                        include_src=args.include_src, git_info=git)
    else:
        run_dirs = sorted(
            d for d in args.experiment_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        print(f"Packaging {len(run_dirs)} runs from {args.experiment_dir}")
        for run_dir in run_dirs:
            print(f"\nRun: {run_dir.name}")
            create_ro_crate(run_dir, args.out,
                            include_src=args.include_src, git_info=git)


if __name__ == "__main__":
    main()
