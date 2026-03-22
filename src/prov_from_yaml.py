#!/usr/bin/env python3
"""
prov_from_yaml.py - Create PROV JSON documents (and optional visualizations)
from simple YAML schema files.

This enables quick schematization of provenance workflows, e.g., for paper
diagrams, without running actual experiments. Define entities, activities,
and relations in YAML, and generate:
  - W3C PROV-compatible JSON
  - DOT graph files (for Graphviz)
  - SVG/PNG images (if Graphviz is installed)

Example YAML schema (see examples/schemas/ for more):
---
prefix:
  ex: "http://example.org/"
  ml: "http://ml-experiment.org/"

entities:
  - id: "ex:dataset_cifar10"
    label: "CIFAR-10 Dataset"
    type: "Dataset"
    attributes:
      samples: 60000
      classes: 10

  - id: "ex:model_cnn"
    label: "CNN Model"
    type: "Model"
    attributes:
      architecture: "Conv2D + FC"
      parameters: 36864

activities:
  - id: "ex:training"
    label: "Training Run"
    attributes:
      epochs: 10
      lr: 0.001
      batch_size: 32

relations:
  - type: "used"
    activity: "ex:training"
    entity: "ex:dataset_cifar10"
  - type: "wasGeneratedBy"
    entity: "ex:model_cnn"
    activity: "ex:training"
---

Usage:
    python src/prov_from_yaml.py --schema schema.yaml --out output.json
    python src/prov_from_yaml.py --schema schema.yaml --out output.json --dot --img
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False


def load_schema(path: Path) -> dict:
    if not YAML_OK:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def schema_to_prov_json(schema: dict) -> dict:
    """Convert a YAML schema dict to a W3C PROV JSON document."""
    prov: Dict[str, Any] = OrderedDict()

    # Prefixes
    prov["prefix"] = schema.get("prefix", {})

    # Entities
    entities = OrderedDict()
    for ent in schema.get("entities", []):
        eid = ent["id"]
        entry = OrderedDict()
        if "label" in ent:
            entry["prov:label"] = ent["label"]
        if "type" in ent:
            entry["prov:type"] = ent["type"]
        for k, v in ent.get("attributes", {}).items():
            entry[k] = v
        entities[eid] = entry
    if entities:
        prov["entity"] = entities

    # Activities
    activities = OrderedDict()
    for act in schema.get("activities", []):
        aid = act["id"]
        entry = OrderedDict()
        if "label" in act:
            entry["prov:label"] = act["label"]
        if "startTime" in act:
            entry["prov:startTime"] = act["startTime"]
        if "endTime" in act:
            entry["prov:endTime"] = act["endTime"]
        for k, v in act.get("attributes", {}).items():
            entry[k] = v
        activities[aid] = entry
    if activities:
        prov["activity"] = activities

    # Agents
    agents = OrderedDict()
    for agent in schema.get("agents", []):
        agid = agent["id"]
        entry = OrderedDict()
        if "label" in agent:
            entry["prov:label"] = agent["label"]
        for k, v in agent.get("attributes", {}).items():
            entry[k] = v
        agents[agid] = entry
    if agents:
        prov["agent"] = agents

    # Relations
    relation_map = {
        "used": [],
        "wasGeneratedBy": [],
        "wasDerivedFrom": [],
        "wasAssociatedWith": [],
        "wasAttributedTo": [],
        "actedOnBehalfOf": [],
    }
    for rel in schema.get("relations", []):
        rtype = rel["type"]
        if rtype not in relation_map:
            relation_map[rtype] = []

        entry = OrderedDict()
        for k, v in rel.items():
            if k != "type":
                entry[f"prov:{k}"] = v
        relation_map[rtype].append(entry)

    for rtype, items in relation_map.items():
        if items:
            prov[rtype] = items

    return prov


def prov_to_dot(prov: dict, title: str = "Provenance Graph") -> str:
    """Generate a Graphviz DOT string from a PROV JSON document."""
    lines = [
        f'digraph "{title}" {{',
        '  rankdir=BT;',
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9];',
        '',
    ]

    # Entities as ellipses (yellow)
    for eid, edata in prov.get("entity", {}).items():
        label = edata.get("prov:label", eid) if isinstance(edata, dict) else eid
        etype = edata.get("prov:type", "") if isinstance(edata, dict) else ""
        display = f"{label}\\n[{etype}]" if etype else label
        safe_id = eid.replace(":", "_").replace("/", "_")
        lines.append(f'  "{safe_id}" [label="{display}", shape=ellipse, style=filled, fillcolor="#FFFFCC"];')

    lines.append('')

    # Activities as rectangles (blue)
    for aid, adata in prov.get("activity", {}).items():
        label = adata.get("prov:label", aid) if isinstance(adata, dict) else aid
        safe_id = aid.replace(":", "_").replace("/", "_")
        lines.append(f'  "{safe_id}" [label="{label}", shape=box, style=filled, fillcolor="#CCE5FF"];')

    lines.append('')

    # Agents as houses (orange)
    for agid, agdata in prov.get("agent", {}).items():
        label = agdata.get("prov:label", agid) if isinstance(agdata, dict) else agid
        safe_id = agid.replace(":", "_").replace("/", "_")
        lines.append(f'  "{safe_id}" [label="{label}", shape=house, style=filled, fillcolor="#FFE0B2"];')

    lines.append('')

    # Relations as edges
    def safe(n):
        return n.replace("prov:", "").replace(":", "_").replace("/", "_")

    for rel in prov.get("used", []):
        act = rel.get("prov:activity", "")
        ent = rel.get("prov:entity", "")
        lines.append(f'  "{safe(act)}" -> "{safe(ent)}" [label="used", style=dashed, color="#666666"];')

    for rel in prov.get("wasGeneratedBy", []):
        ent = rel.get("prov:entity", "")
        act = rel.get("prov:activity", "")
        lines.append(f'  "{safe(ent)}" -> "{safe(act)}" [label="wasGeneratedBy", color="#2196F3"];')

    for rel in prov.get("wasDerivedFrom", []):
        derived = rel.get("prov:generatedEntity", rel.get("prov:entity", ""))
        source = rel.get("prov:usedEntity", rel.get("prov:from", ""))
        lines.append(f'  "{safe(derived)}" -> "{safe(source)}" [label="wasDerivedFrom", color="#4CAF50"];')

    for rel in prov.get("wasAssociatedWith", []):
        act = rel.get("prov:activity", "")
        agent = rel.get("prov:agent", "")
        lines.append(f'  "{safe(act)}" -> "{safe(agent)}" [label="wasAssociatedWith", color="#FF9800"];')

    for rel in prov.get("wasAttributedTo", []):
        ent = rel.get("prov:entity", "")
        agent = rel.get("prov:agent", "")
        lines.append(f'  "{safe(ent)}" -> "{safe(agent)}" [label="wasAttributedTo", color="#9C27B0"];')

    lines.append('}')
    return '\n'.join(lines)


def render_dot(dot_content: str, output_path: Path, fmt: str = "svg"):
    """Render a DOT string to an image using Graphviz."""
    dot_path = output_path.with_suffix(".dot")
    dot_path.write_text(dot_content, encoding="utf-8")

    img_path = output_path.with_suffix(f".{fmt}")
    try:
        subprocess.run(
            ["dot", f"-T{fmt}", str(dot_path), "-o", str(img_path)],
            check=True, capture_output=True,
        )
        print(f"  Image: {img_path}")
    except FileNotFoundError:
        print("  Graphviz 'dot' not found. Install Graphviz to render images.")
        print(f"  DOT file saved: {dot_path}")
    except subprocess.CalledProcessError as e:
        print(f"  Graphviz rendering failed: {e.stderr.decode()}")


def main():
    parser = argparse.ArgumentParser(
        description="Create PROV JSON and visualizations from YAML schemas."
    )
    parser.add_argument("--schema", type=Path, required=True,
                        help="Path to the YAML schema file")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path for the PROV JSON file")
    parser.add_argument("--dot", action="store_true",
                        help="Also generate a DOT graph file")
    parser.add_argument("--img", action="store_true",
                        help="Also render to SVG image (requires Graphviz)")
    parser.add_argument("--img-format", type=str, default="svg",
                        choices=["svg", "png", "pdf"],
                        help="Image format (default: svg)")
    parser.add_argument("--title", type=str, default="Provenance Graph",
                        help="Title for the graph visualization")
    args = parser.parse_args()

    if not args.schema.exists():
        raise SystemExit(f"Schema file not found: {args.schema}")

    schema = load_schema(args.schema)
    prov = schema_to_prov_json(schema)

    # Write JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, default=str)
    print(f"  PROV JSON: {args.out}")

    # DOT / Image
    if args.dot or args.img:
        dot_content = prov_to_dot(prov, title=args.title)

        if args.dot:
            dot_path = args.out.with_suffix(".dot")
            dot_path.write_text(dot_content, encoding="utf-8")
            print(f"  DOT file: {dot_path}")

        if args.img:
            render_dot(dot_content, args.out, fmt=args.img_format)


if __name__ == "__main__":
    main()
