# src/prov_to_csv.py - Updated to handle both CSV and NetCDF files
import argparse
import re
from pathlib import Path
import pandas as pd
from prov.model import ProvDocument, QualifiedName, ProvActivity, ProvEntity
import numpy as np

BASE_ROOT = Path("data/prov/prov.jsonl")
OUT_DIR = Path("data/unified")
DEBUG = False

# ----------------------------- Optional deps -----------------------------
try:
    import xarray as xr
    XARRAY_OK = True
except ImportError:
    XARRAY_OK = False
    print("⚠️  Warning: xarray not installed. NetCDF files (.nc) will be skipped.")
    print("   Install with: pip install xarray netCDF4")

# -------------------------- utils --------------------------
def dbg(msg: str):
    if DEBUG:
        print(msg)

def qn_to_prefixed(qn) -> str:
    if isinstance(qn, QualifiedName):
        pref = qn.namespace.prefix if qn.namespace else None
        local = qn.localpart
        return f"{pref}:{local}" if pref else local
    return str(qn)

def normalize_key(prefixed: str) -> str:
    """
    Normalize PROV attribute keys while preserving structure.
    
    Rules:
    - yProv4ML:* → keep as-is (e.g., yProv4ML:level)
    - metric:X → X (strip metric: prefix)
    - param:X → param_X
    - context:X → context_X
    - Plain names (MODEL_SIZE, accuracy, etc.) → keep as-is
    - Other namespaces → replace : with _
    """
    # Already a plain name (no namespace)
    if ":" not in prefixed:
        return prefixed
    
    # Keep yProv4ML namespace as-is
    if prefixed.startswith("yProv4ML:"):
        return prefixed
    
    # Strip metric: prefix
    if prefixed.startswith("metric:"):
        return prefixed.split(":", 1)[1]
    
    # Convert param: to param_
    if prefixed.startswith("param:"):
        return "param_" + prefixed.split(":", 1)[1]
    
    # Convert context: to context_
    if prefixed.startswith("context:"):
        return "context_" + prefixed.split(":", 1)[1]
    
    # For other namespaces, replace : with _
    return prefixed.replace(":", "_")

def clean_scalar_for_csv(x):
    """
    Make sure IDs / version strings can't break CSV structure.

    - Convert to string
    - Collapse whitespace/newlines
    - Replace commas with semicolons
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x)
    # collapse all whitespace (spaces, newlines, tabs) into single spaces
    s = " ".join(s.split())
    # avoid introducing extra CSV columns via commas
    s = s.replace(",", ";")
    return s


# ----------------------- metric reading -----------------------
def read_metric_csv(csv_path: Path, metric_name: str | None = None):
    """Read CSV metric file and return last value."""
    try:
        if not csv_path.exists():
            dbg(f"   ❓ Missing CSV: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        if df.empty:
            dbg(f"   ⚠️ Empty CSV: {csv_path}")
            return None
        
        # Get last row, first numeric column
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                val = float(df[col].iloc[-1])
                dbg(f"   ✓ {csv_path.name}: {col} = {val:.6g}")
                return val
        
        dbg(f"   ⚠️ No numeric columns in {csv_path}")
        return None
    except Exception as e:
        print(f"⚠️  Could not read CSV {csv_path}: {e}")
        return None

def read_metric_nc(nc_path: Path, metric_name: str | None = None):
    """Read NetCDF metric file and return last value."""
    if not XARRAY_OK:
        dbg(f"   ⚠️ xarray not installed, skipping {nc_path}")
        return None
    
    try:
        if not nc_path.exists():
            dbg(f"   ❓ Missing NetCDF: {nc_path}")
            return None
        
        ds = xr.open_dataset(nc_path)
        data_vars = list(ds.data_vars)
        
        if not data_vars:
            dbg(f"   ⚠️ No data variables in {nc_path}")
            ds.close()
            return None
        
        # CRITICAL FIX: Prefer 'values' variable if it exists (that's where the actual metric is)
        # Otherwise fall back to first variable
        var_name = 'values' if 'values' in data_vars else data_vars[0]
        
        values = ds[var_name].values.flatten()
        ds.close()
        
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            dbg(f"   ⚠️ No valid values in {nc_path}")
            return None
        
        val = float(valid[-1])
        dbg(f"   ✓ {nc_path.name}: {var_name} = {val:.6g}")
        return val
    
    except Exception as e:
        print(f"⚠️  Could not read NetCDF {nc_path}: {e}")
        return None

def read_metric_file(file_path: Path, metric_name: str | None = None):
    """Dispatch to appropriate reader based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == '.csv':
        return read_metric_csv(file_path, metric_name)
    elif suffix == '.nc':
        return read_metric_nc(file_path, metric_name)
    else:
        dbg(f"   ⚠️ Unknown file type: {file_path}")
        return None

# -------------------- PROV extraction --------------------
def load_prov_json(s: Path) -> ProvDocument:
    return ProvDocument.deserialize(content=s, format="json")

def extract_metrics_from_entities(prov_doc: ProvDocument) -> dict:
    """Extract metric values by reading metric files (CSV or NetCDF)."""
    metrics = {}
    
    # CRITICAL: Use the JSON file's actual parent directory
    # e.g., if JSON is at data/prov/usecase/exp_0/prov_file.json
    # then base_dir is data/prov/usecase/exp_0
    metric_count = 0
    for rec in prov_doc.get_records():
        if not isinstance(rec, ProvEntity):
            continue

        attrs = dict(rec.attributes)

        # --- pull out common attributes ---
        metric_name = None
        metric_path = None
        source_val = None
        role_val = None

        for k, v in attrs.items():
            k_str = qn_to_prefixed(k)
            if k_str == "yProv4ML:label":
                metric_name = str(v)
            elif k_str == "yProv4ML:path":
                metric_path = str(v)
            elif k_str == "yProv4ML:source":
                source_val = str(v)
            elif k_str == "yProv4ML:role":
                role_val = str(v)

        # Need at least a path
        if not metric_path:
            continue

        # --- decide if this entity is a metric ---
        is_metric = False

        # 1) Based on source field (when present)
        if source_val and "METRIC" in source_val.upper():
            is_metric = True

        # 2) Based on path/location: anything under a metrics dir with .nc/.csv
        lower_path = metric_path.lower()
        if ("/metrics" in lower_path or "metrics_" in lower_path) and (
            lower_path.endswith(".nc") or lower_path.endswith(".csv")
        ):
            is_metric = True

        # 3) Skip obvious pure inputs unless already tagged as metric
        if role_val and role_val.lower() == "input":
            if not is_metric:
                continue

        if not is_metric:
            continue

        # Fallback metric name from filename if label missing
        if not metric_name:
            metric_name = Path(metric_path).stem

        metric_count += 1
        dbg(f"   📊 Metric entity: {metric_name} -> {metric_path}")

        # --- FIXED path resolution strategy ---
        file_path = None
        candidates = []

        base_dir = Path.cwd()
        p = Path(metric_path)
        
        if p.is_absolute():
            candidates.append(p)
        else:
            # KEY FIX: If path starts with "prov/", it's relative to the experiment folder
            # Example: JSON at data/prov/usecase/exp_0/prov_file.json
            #          Metric path: prov/exp_0/metrics_GR0/emissions.nc
            #          Should resolve to: data/prov/usecase/exp_0/metrics_GR0/emissions.nc
            if metric_path.startswith("prov/"):
                # Extract the part after "prov/"
                rel_part = metric_path[5:]  # Remove "prov/" prefix
                
                # Strategy 1: Relative to JSON file's parent (same experiment folder)
                # If JSON is at .../usecase/exp_0/prov_file.json and metric_path is prov/exp_0/metrics/...
                # Then base_dir / rel_part gives .../usecase/exp_0/metrics/...
                candidates.append(base_dir / rel_part)
                
                # Strategy 2: Go up one level from JSON parent (for some structures)
                candidates.append(base_dir.parent / rel_part)
                
            # Strategy 3: Path relative to JSON location (as-is)
            candidates.append(base_dir / metric_path)
            
            # Strategy 4: Relative to current working directory
            candidates.append(Path.cwd() / metric_path)
            
            # Strategy 5: Prepend 'data/' and try from CWD
            candidates.append(Path.cwd() / "data" / metric_path)

        # Find the first path that exists
        for candidate in candidates:
            if candidate.exists():
                file_path = candidate
                dbg(f"      ✓ Found metric file at: {file_path}")
                break

        if file_path is None:
            dbg(f"      ✗ Metric file not found for {metric_name}. Tried:")
            for c in candidates[:5]:
                dbg(f"         - {c}")
            continue

        value = read_metric_file(file_path, metric_name=metric_name)
        if value is not None:
            metrics[metric_name] = value
            dbg(f"      ✓ Extracted metric {metric_name} = {value}")
        else:
            dbg(f"      ✗ Failed to extract value from {file_path}")

    dbg(f"   Total: Found {metric_count} metric entities, extracted {len(metrics)} values")
    return metrics


def prov_activity_to_row(activity: ProvActivity, prov_doc: ProvDocument, prov_json : str):
    """
    Extract ALL data from PROV document:
    - Main activity attributes
    - Sub-activity attributes (Context.TRAINING, Context.TESTING, etc.)
    - Metric values from NetCDF/CSV files
    - Artifacts
    """
    row = {}
    
    # 1. Extract main activity attributes
    attrs = dict(activity.attributes)
    for k, v in attrs.items():
        k_str = qn_to_prefixed(k)
        row[normalize_key(k_str)] = v
    
    # 2. Extract ALL sub-activity attributes (Context.TRAINING, etc.)
    for rec in prov_doc.get_records():
        if isinstance(rec, ProvActivity) and rec != activity:
            sub_attrs = dict(rec.attributes)
            # Get the context name
            activity_id = str(rec.identifier)
            context_name = activity_id.split(":")[-1] if ":" in activity_id else activity_id
            
            for k, v in sub_attrs.items():
                k_str = qn_to_prefixed(k)
                # Skip yProv4ML:level (internal metadata)
                if k_str == "yProv4ML:level":
                    continue
                
                # For context-specific attributes, use plain column name
                col_name = normalize_key(k_str)
                row[col_name] = v
    
    # 3. Extract artifacts
    arts = []
    for rec in prov_doc.get_records():
        if isinstance(rec, ProvEntity):
            ent_attrs = dict(rec.attributes)
            for k2, v2 in ent_attrs.items():
                if qn_to_prefixed(k2) == "yProv4ML:path" and "artifact" in str(v2).lower():
                    arts.append(str(v2))
                    break
    if arts:
        row["artifacts"] = ";".join(sorted(set(arts)))
    
    # 4. Extract metric values from NetCDF/CSV files
    metrics = extract_metrics_from_entities(prov_doc)
    row.update(metrics)
    
    # 5. Set experiment identifiers (SANITIZED)
    exp_raw = (
        row.get("yProv4ML:experiment_name")
        or row.get("context_experiment_name")
        or row.get("experiment_name")
    )
    run_raw = (
        row.get("yProv4ML:run_id")
        or row.get("context_run_id")
        or row.get("run_id")
    )

    exp_clean = clean_scalar_for_csv(exp_raw)
    run_clean = clean_scalar_for_csv(run_raw)

    if exp_clean is not None:
        row["exp"] = exp_clean

    if run_clean is not None:
        # ensure we keep a clean, consistent run_id
        row["run_id"] = run_clean
        row["yProv4ML:run_id"] = run_clean

    # 6. Clean python version as well (SANITIZED)
    if "yProv4ML:python_version" in row:
        row["yProv4ML:python_version"] = clean_scalar_for_csv(
            row["yProv4ML:python_version"]
        )

    return row

def iter_all_prov_lines(input_file: Path):
    with open(input_file.resolve(), "r") as f: 
        stripped_lines = lambda f : (l.rstrip("\n") for l in f)
        for l in stripped_lines(f):
            yield l

def infer_exp_version(prov_doc: ProvDocument):
    exp_raw, run_raw = None, None
    for rec in prov_doc.get_records():
        if isinstance(rec, ProvActivity):
            attrs = dict(rec.attributes)
            for k, v in attrs.items():
                k_str = qn_to_prefixed(k).strip()
                if k_str == "yProv4ML:experiment_name": 
                    exp_raw = v
                if k_str == "yProv4ML:run_id": 
                    run_raw = v

                if exp_raw is not None and run_raw is not None: 
                    return exp_raw, run_raw
    raise Exception("No exp name or run_id found") 

def safe_filename(name: str) -> str:
    if name is None or str(name).strip() == "":
        return "unknown_experiment"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")

def build_dataframe(input_file: Path) -> pd.DataFrame | None:
    rows = []
    for prov_json in iter_all_prov_lines(input_file):
        dbg(f"\n📄 Processing: {prov_json}")

        d = load_prov_json(prov_json)
        exp_name, exp_version = infer_exp_version(d)
        activities = [r for r in d.get_records() if isinstance(r, ProvActivity)]
        if not activities:
            dbg(f"   ⚠️ No activities found")
            continue

        a = activities[0]
        row = prov_activity_to_row(a, d, prov_json)

        if not row.get("exp") and exp_name:
            row["exp"] = exp_name

        row["exp_folder"] = exp_name
        row["exp_version"] = exp_version
        row["source_file"] = str(prov_json)
        rows.append(row)

    if not rows:
        return None

    # Create DataFrame - pandas will handle columns dynamically
    df = pd.DataFrame(rows)
    
    # Optional: organize column order (move common columns to front, keep rest as-is)
    priority_cols = ["exp", "run_id", "exp_folder", "exp_version"]
    existing_priority = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in priority_cols]
    
    if existing_priority:
        df = df[existing_priority + other_cols]
    
    return df

def write_split_by_exp(df: pd.DataFrame, out_dir: Path):
    """
    Write CSV files per experiment, with each file containing ONLY the columns
    that have data for that specific experiment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    key = "exp" if "exp" in df.columns else "exp_folder"
    file_count = 0

    print(f"\n Saving whole file ({len(df)} total runs) in unified.csv...")
    df.to_csv(out_dir/"unified.csv")

    print(f"\n📊 Processing {len(df)} total runs across experiments...")
    print()

    for exp_value, g in df.groupby(key, dropna=False, sort=True):
        stem = safe_filename(exp_value if pd.notna(exp_value) else "unknown")
        out_path = out_dir / f"{stem}.csv"
        
        # Drop columns that are completely empty for this experiment
        non_empty_cols = [c for c in g.columns if g[c].notna().any()]
        g_clean = g[non_empty_cols].copy()
        
        # Write CSV
        g_clean.to_csv(out_path, index=False)
        file_count += 1
        
        # Show summary
        print(f"✅ {out_path.name}")
        print(f"   Runs: {len(g_clean)}")
        print(f"   Columns: {len(g_clean.columns)} (removed {len(g.columns) - len(g_clean.columns)} empty columns)")
        
        # Show column breakdown
        yprov_cols = [c for c in g_clean.columns if str(c).startswith("yProv4ML:")]
        param_cols = [c for c in g_clean.columns if str(c).startswith("param_")]
        metric_cols = [c for c in g_clean.columns if c not in yprov_cols + param_cols 
                       and c not in ["exp", "run_id", "exp_folder", "exp_version", "source_file", "artifacts"]
                       and not str(c).startswith("context_")]
        
        if yprov_cols:
            print(f"   • yProv4ML: {len(yprov_cols)} attributes")
        if param_cols:
            print(f"   • Parameters: {len(param_cols)} ({', '.join(param_cols)})")
        if metric_cols:
            print(f"   • Metrics: {len(metric_cols)} ({', '.join(sorted(metric_cols))})")
        print()

    print(f"✅ Created {file_count} CSV file(s) in {out_dir}")
    print(f"   Each CSV contains only the columns with data for that experiment")

def main(input_file: Path, out_dir: Path):
    print(f"📂 Processing PROV file at: {input_file}")
    print(f"📤 Output directory: {out_dir}")
    print()

    df = build_dataframe(input_file)
    if df is None:
        print(f"❌ No runs/activities found in PROV JSON file at {input_file}.")
        return

    print(f"✅ Found {len(df)} runs total")
    write_split_by_exp(df, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate PROV JSONs into per-experiment CSVs.")
    parser.add_argument("--input_file", type=Path, default=BASE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    DEBUG = args.debug
    main(args.input_file, args.out_dir)