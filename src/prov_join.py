#!/usr/bin/env python3
"""
prov_join.py - Join multiple PROV JSON files into a unified provenance document.

Merges provenance from multiple experiment runs into a single JSON file,
de-duplicating entities and activities by their identifiers. This is useful for:
  - Combining runs from the same experiment into one provenance graph
  - Joining experiments across different configurations for comparison
  - Building a unified view of multi-process provenance (e.g., when GR0 changes)

Inspired by the y2Graph library (https://github.com/HPCI-Lab/y2Graph) approach
to joining JSON provenance documents based on entity/activity IDs.

Usage:
    # Join all prov JSONs under a directory into one file
    python src/prov_join.py --root data/prov/my_experiment --out merged_prov.json

    # Join specific files
    python src/prov_join.py --files run1/prov_0.json run2/prov_0.json --out merged.json

    # Join and keep per-source tags (for traceability)
    python src/prov_join.py --root data/prov --out merged.json --tag-sources

    # Join multi-process provenance (GR0, GR1, ...) with dedup
    python src/prov_join.py --root data/prov --out merged.json --multi-process
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def load_prov_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _entity_id(entity_key: str, entity_data: dict) -> str:
    """Compute a stable ID for deduplication.

    Priority: yProv4ML:label > yProv4ML:path > the JSON key itself.
    """
    if isinstance(entity_data, dict):
        label = entity_data.get("yProv4ML:label") or entity_data.get("prov:label")
        path = entity_data.get("yProv4ML:path")
        if label:
            return str(label)
        if path:
            return str(path)
    return entity_key


def _activity_id(activity_key: str, activity_data: dict) -> str:
    """Compute a stable ID for activity deduplication."""
    if isinstance(activity_data, dict):
        run_id = activity_data.get("yProv4ML:run_id")
        exp = activity_data.get("yProv4ML:experiment_name")
        if run_id:
            return str(run_id)
        if exp:
            return f"{exp}:{activity_key}"
    return activity_key


def _merge_dict_section(
    target: dict,
    source: dict,
    seen_ids: Set[str],
    id_fn,
    source_tag: Optional[str] = None,
) -> int:
    """Merge a PROV section (entity, activity, etc.) into target, skipping duplicates.

    Returns the number of new items added.
    """
    added = 0
    for key, value in source.items():
        stable_id = id_fn(key, value)
        if stable_id in seen_ids:
            continue
        seen_ids.add(stable_id)

        # If the key already exists in target, make it unique
        out_key = key
        if out_key in target:
            suffix = 1
            while f"{out_key}_{suffix}" in target:
                suffix += 1
            out_key = f"{out_key}_{suffix}"

        entry = copy.deepcopy(value) if isinstance(value, dict) else value
        if source_tag and isinstance(entry, dict):
            entry["_source_file"] = source_tag

        target[out_key] = entry
        added += 1
    return added


def _merge_relation_list(
    target: list,
    source: list,
    seen: Set[str],
) -> int:
    """Merge relation arrays (wasGeneratedBy, used, etc.), deduplicating by content hash."""
    added = 0
    for item in source:
        item_key = json.dumps(item, sort_keys=True, default=str)
        if item_key in seen:
            continue
        seen.add(item_key)
        target.append(copy.deepcopy(item))
        added += 1
    return added


def join_prov_documents(
    docs: List[dict],
    source_paths: Optional[List[str]] = None,
    tag_sources: bool = False,
    multi_process: bool = False,
) -> dict:
    """Join multiple PROV JSON documents into one unified document.

    Args:
        docs: List of parsed PROV JSON dicts.
        source_paths: Optional list of file paths (same length as docs) for tagging.
        tag_sources: If True, add _source_file attribute to merged entities/activities.
        multi_process: If True, use relaxed deduplication for multi-process provenance
                       (e.g., metrics from GR0 vs GR1 are kept separately).

    Returns:
        Merged PROV JSON dict.
    """
    merged: Dict[str, Any] = OrderedDict()
    merged["prefix"] = {}
    merged["entity"] = OrderedDict()
    merged["activity"] = OrderedDict()
    merged["wasGeneratedBy"] = []
    merged["used"] = []
    merged["wasDerivedFrom"] = []
    merged["wasAssociatedWith"] = []
    merged["wasAttributedTo"] = []

    entity_ids: Set[str] = set()
    activity_ids: Set[str] = set()
    relation_seen: Dict[str, Set[str]] = {
        "wasGeneratedBy": set(),
        "used": set(),
        "wasDerivedFrom": set(),
        "wasAssociatedWith": set(),
        "wasAttributedTo": set(),
    }

    stats = {"files": 0, "entities_added": 0, "entities_skipped": 0,
             "activities_added": 0, "activities_skipped": 0, "relations_added": 0}

    for i, doc in enumerate(docs):
        source_tag = None
        if tag_sources and source_paths and i < len(source_paths):
            source_tag = source_paths[i]

        stats["files"] += 1

        # Merge prefixes
        if "prefix" in doc and isinstance(doc["prefix"], dict):
            merged["prefix"].update(doc["prefix"])

        # Merge entities
        if "entity" in doc and isinstance(doc["entity"], dict):
            eid_fn = _entity_id
            if multi_process:
                # In multi-process mode, prefix entity IDs with source to avoid
                # collapsing metrics from different processes (GR0 vs GR1)
                process_tag = _detect_process_tag(doc, source_paths[i] if source_paths else None)
                def eid_fn(k, v, _tag=process_tag):
                    base = _entity_id(k, v)
                    return f"{_tag}:{base}" if _tag else base

            before = len(entity_ids)
            n = _merge_dict_section(merged["entity"], doc["entity"], entity_ids, eid_fn, source_tag)
            stats["entities_added"] += n
            stats["entities_skipped"] += len(doc["entity"]) - n

        # Merge activities
        if "activity" in doc and isinstance(doc["activity"], dict):
            before = len(activity_ids)
            n = _merge_dict_section(merged["activity"], doc["activity"], activity_ids, _activity_id, source_tag)
            stats["activities_added"] += n
            stats["activities_skipped"] += len(doc["activity"]) - n

        # Merge relations
        for rel_key in relation_seen:
            if rel_key in doc:
                items = doc[rel_key]
                if isinstance(items, list):
                    n = _merge_relation_list(merged[rel_key], items, relation_seen[rel_key])
                    stats["relations_added"] += n
                elif isinstance(items, dict):
                    # Some PROV serializations use dict format for relations
                    as_list = [{"_key": k, **v} if isinstance(v, dict) else {"_key": k, "_value": v}
                               for k, v in items.items()]
                    n = _merge_relation_list(merged[rel_key], as_list, relation_seen[rel_key])
                    stats["relations_added"] += n

    # Clean up empty sections
    for key in list(merged.keys()):
        val = merged[key]
        if isinstance(val, (dict, list)) and len(val) == 0:
            del merged[key]

    merged["_join_stats"] = stats
    return merged


def _detect_process_tag(doc: dict, source_path: Optional[str] = None) -> Optional[str]:
    """Detect the process rank/tag (e.g., GR0, GR1) from a PROV document or its path."""
    # Check path for metrics_GR0, metrics_GR1 patterns
    if source_path:
        import re
        m = re.search(r'GR(\d+)', source_path)
        if m:
            return f"GR{m.group(1)}"

    # Check entity paths within the document
    if "entity" in doc and isinstance(doc["entity"], dict):
        import re
        for key, val in doc["entity"].items():
            if isinstance(val, dict):
                path = val.get("yProv4ML:path", "")
                m = re.search(r'metrics_GR(\d+)', str(path))
                if m:
                    return f"GR{m.group(1)}"

    return None


def find_prov_jsons(root: Path) -> List[Path]:
    """Recursively find all prov_*.json files under root."""
    return sorted(Path(p) for p in glob.iglob(str(root / "**" / "prov_*.json"), recursive=True))


def main():
    parser = argparse.ArgumentParser(
        description="Join multiple PROV JSON files into a unified provenance document."
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Root directory to search for prov_*.json files")
    parser.add_argument("--files", nargs="+", type=Path, default=None,
                        help="Specific PROV JSON files to join")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path for the merged PROV JSON")
    parser.add_argument("--tag-sources", action="store_true",
                        help="Add _source_file attribute to track provenance of each element")
    parser.add_argument("--multi-process", action="store_true",
                        help="Enable multi-process dedup (keeps GR0/GR1 metrics separate)")
    parser.add_argument("--pretty", action="store_true", default=True,
                        help="Pretty-print the output JSON (default: True)")
    args = parser.parse_args()

    # Collect input files
    if args.files:
        prov_files = [p for p in args.files if p.exists()]
    elif args.root:
        prov_files = find_prov_jsons(args.root)
    else:
        parser.error("Specify either --root or --files")

    if not prov_files:
        print("No PROV JSON files found.")
        return

    print(f"Found {len(prov_files)} PROV JSON file(s)")

    # Load all documents
    docs = []
    paths = []
    for p in prov_files:
        try:
            docs.append(load_prov_json(p))
            paths.append(str(p))
            print(f"  Loaded: {p}")
        except Exception as e:
            print(f"  Failed to load {p}: {e}")

    # Join
    merged = join_prov_documents(
        docs,
        source_paths=paths,
        tag_sources=args.tag_sources,
        multi_process=args.multi_process,
    )

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2 if args.pretty else None, default=str)

    stats = merged.get("_join_stats", {})
    print(f"\nMerged {stats.get('files', 0)} files -> {args.out}")
    print(f"  Entities: {stats.get('entities_added', 0)} added, {stats.get('entities_skipped', 0)} deduplicated")
    print(f"  Activities: {stats.get('activities_added', 0)} added, {stats.get('activities_skipped', 0)} deduplicated")
    print(f"  Relations: {stats.get('relations_added', 0)} added")


if __name__ == "__main__":
    main()
