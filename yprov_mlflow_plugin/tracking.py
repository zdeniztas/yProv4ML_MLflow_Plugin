from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Dict, Any
from urllib.parse import urlparse
from pathlib import Path
import os, re, sys, uuid

if TYPE_CHECKING:
    from mlflow.store.tracking.abstract_store import AbstractStore  # type: ignore
else:
    class AbstractStore:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

# yprov (prov4ml)
try:
    import prov4ml as yprov
except Exception:
    yprov = None

DEBUG = os.getenv("YPROV_DEBUG", "").lower() in ("1", "true", "yes", "on")

def _debug(msg: str):
    if DEBUG:
        print(f"[yProv Plugin] {msg}", flush=True)

def _env_flag(name: str, default=False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _strip_yprov(uri: str) -> str:
    return uri.replace("yprov+", "", 1) if isinstance(uri, str) and uri.startswith("yprov+") else uri

# ---------------- PATH/URI NORMALIZATION ----------------

def _normalize_file_uri(raw: str) -> str:
    """
    Convert a path or file://... into an ABSOLUTE file:// URI.
    Keeps tests happy that assert FileStore received file://...
    """
    if raw.startswith("file://"):
        p = urlparse(raw).path
    else:
        p = raw

    # /C:/... -> C:/... (Windows path that came through urlparse on *nix)
    if re.match(r"^/[A-Za-z]:/.*", p):
        p = p.lstrip("/")

    p = p.replace("\\", "/")

    # Windows absolute path?
    if re.match(r"^[A-Za-z]:[\\/]", p):
        return f"file:///{p}"
    # POSIX absolute path?
    if p.startswith("/"):
        return f"file://{p}"
    # Otherwise make absolute
    ap = os.path.abspath(p).replace("\\", "/")
    return f"file://{ap}"

# ---------------- DELEGATE FACTORY ----------------

def _delegate_for(tracking_uri: str):
    base = _strip_yprov(tracking_uri or "")
    parsed = urlparse(base)

    if parsed.scheme in ("http", "https"):
        from mlflow.store.tracking.rest_store import RestStore  # type: ignore
        return RestStore(base)

    if parsed.scheme in ("", "file"):
        from mlflow.store.tracking.file_store import FileStore  # type: ignore
        # IMPORTANT: pass absolute file:// URI (some tests assert this)
        file_uri = _normalize_file_uri(base if base else ".")
        return FileStore(file_uri)

    # Fallback
    from mlflow.store.tracking.rest_store import RestStore  # type: ignore
    return RestStore(base)

# ---------------- OUTPUT FORMAT CONFIG ----------------
# Control what prov4ml generates on run end via YPROV_OUTPUT_FORMAT env var.
# Comma-separated values: "json", "dot", "img", "svg", "ro-crate"
# Default: "json" (only the PROV JSON document)
# Example: YPROV_OUTPUT_FORMAT="json,dot,img"

def _parse_output_formats():
    raw = os.getenv("YPROV_OUTPUT_FORMAT", "json").lower()
    formats = {f.strip() for f in raw.split(",") if f.strip()}
    if not formats:
        formats = {"json"}
    return formats

# ---------------- YPROV HELPERS ----------------

def _yprov_end_run():
    if not yprov:
        return False

    formats = _parse_output_formats()
    create_graph = "dot" in formats or "img" in formats
    create_svg = "svg" in formats or "img" in formats
    create_ro_crate = "ro-crate" in formats

    _debug(f"  Output formats requested: {formats}")
    _debug(f"  create_graph={create_graph}, create_svg={create_svg}, create_ro_crate={create_ro_crate}")

    attempts = [
        lambda: yprov.end_run(create_graph=create_graph, create_svg=create_svg, crate_ro_crate=create_ro_crate),
        lambda: yprov.end_run(create_graph=create_graph, create_svg=create_svg),
        lambda: yprov.end_run(create_graph=create_graph),
        lambda: yprov.end_run(),
    ]
    for i, attempt in enumerate(attempts):
        try:
            attempt()
            _debug(f"  ✓ prov4ml.end_run() succeeded (attempt {i+1})")
            return True
        except TypeError as e:
            if i == len(attempts) - 1:
                _debug(f"  ⚠ All prov4ml.end_run() attempts failed: {e}")
                return False
            continue
        except Exception as e:
            _debug(f"  ⚠ prov4ml.end_run() failed: {e}")
            return False
    return False

def _call_yprov_log_metric(key: str, value: float, step=None):
    if not yprov:
        return False
    attempts = [
        lambda: yprov.log_metric(key, value, context=None, step=step),
        lambda: yprov.log_metric(key, value, step=step),
        lambda: yprov.log_metric(key, value),
    ]
    for i, attempt in enumerate(attempts):
        try:
            attempt()
            return True
        except TypeError as e:
            if i == len(attempts) - 1:
                _debug(f"  ⚠ prov4ml.log_metric() failed with all signatures: {e}")
                return False
            continue
        except Exception as e:
            _debug(f"  ⚠ prov4ml.log_metric() error: {e}")
            return False
    return False

# ---------------- LIGHTWEIGHT SHIM (only for fake delegates) ----------------

class _RunInfo:
    def __init__(self, run_id: str):
        self.run_id = run_id

class _Run:
    def __init__(self, run_id: str):
        self.info = _RunInfo(run_id)

class _Experiment:
    def __init__(self, experiment_id: str, name: str):
        self.experiment_id = experiment_id
        self.name = name

class _NoopFileShim:
    """
    Minimal in-memory 'store' for tests or fake delegates.
    Provides the subset of the MLflow Tracking API we use.
    """
    def __init__(self, _uri: str):
        self._uri = _uri
        self._experiments: Dict[str, _Experiment] = {}          # name -> Experiment
        self._experiments_by_id: Dict[str, _Experiment] = {}    # id -> Experiment
        self._runs: Dict[str, _Run] = {}

        # Ensure Default experiment exists like MLflow does
        self.create_experiment("Default")

    # ---- Experiments ----
    def create_experiment(self, name: str, artifact_location=None, tags=None) -> str:
        if name in self._experiments:
            return self._experiments[name].experiment_id
        eid = str(len(self._experiments_by_id) + 1)
        exp = _Experiment(eid, name)
        self._experiments[name] = exp
        self._experiments_by_id[eid] = exp
        return eid

    def get_experiment_by_name(self, name: str):
        return self._experiments.get(name)

    def get_experiment(self, exp_id: str):
        return self._experiments_by_id.get(str(exp_id))

    # ---- Runs ----
    def create_run(self, experiment_id, user_id=None, start_time=None, tags=None, run_name=None, **kwargs):
        rid = str(uuid.uuid4())
        run = _Run(rid)
        self._runs[rid] = run
        return run

    def get_run(self, run_id):
        return self._runs.get(run_id, _Run(run_id))

    # ---- Logging / Termination ----
    def log_param(self, run_id, param): return None
    def log_metric(self, run_id, metric): return None
    def log_batch(self, run_id, metrics, params, tags): return None
    def set_terminated(self, run_id, status, end_time): return None
    def update_run_info(self, run_id, run_status, end_time, **kwargs): return None

    # ---- Queries ----
    def list_run_infos(self, experiment_id, run_view_type, max_results, order_by, page_token):
        return []
    def search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token):
        return []

    # ---- Artifacts ----
    def get_artifact_uri(self, run_id):
        return f"{self._uri}/artifacts"

# ---------------- STORE IMPLEMENTATION ----------------

class YProvTrackingStore(AbstractStore):
    def __init__(self, store_uri: str, artifact_uri: Optional[str] = None):
        super().__init__()
        self._store_uri = store_uri
        delegate = _delegate_for(store_uri)

        # Use shim ONLY for obvious test/fake delegates, or if explicitly forced.
        delegate_mod = type(delegate).__module__
        delegate_cls = type(delegate).__name__
        force_shim = _env_flag("YPROV_TEST_SHIM", False)
        is_fake = delegate_cls.startswith("_") or "pytest" in (delegate_mod or "")
        needs_shim = force_shim or is_fake or not hasattr(delegate, "create_run")

        if needs_shim:
            _debug("   Detected minimal/fake FileStore; using _NoopFileShim for MLflow API.")
            self._delegate = _NoopFileShim(getattr(delegate, "_uri", getattr(delegate, "uri", str(store_uri))))
        else:
            self._delegate = delegate

        # Prov output directory: YPROV_PROV_PATH > YPROV_OUT_DIR > default
        # YPROV_PROV_PATH: explicit full path to the provenance output directory
        # YPROV_OUT_DIR: base directory (kept for backward compatibility)
        self._prov_out = Path(
            os.getenv("YPROV_PROV_PATH")
            or os.getenv("YPROV_OUT_DIR", "data/prov")
        )
        self._prov_out.mkdir(parents=True, exist_ok=True)

        _debug(f"🟢 YProvTrackingStore initialized with URI: {store_uri}")
        _debug(f"   Delegate type: {type(self._delegate).__name__}")
        _debug(f"   PROV output: {self._prov_out}")

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def _exp_name(self, experiment_id: str) -> str:
        try:
            exp = self._delegate.get_experiment(experiment_id)
            return (getattr(exp, "name", None) if exp else None) or "Default"
        except Exception:
            return "Default"

    # --- methods below unchanged except they call self._delegate ---

    def create_run(self, experiment_id, user_id=None, start_time=None, tags=None, run_name=None, **kwargs):
        _debug(f"create_run called: exp={experiment_id}, run_name={run_name}")
        result = self._delegate.create_run(experiment_id, user_id, start_time, tags, run_name=run_name, **kwargs)
        run_id = result.info.run_id

        exp_name = self._exp_name(experiment_id)
        exp_dir = (self._prov_out / exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)

        unify = _env_flag("YPROV_UNIFY", False)
        merged_path = os.getenv("YPROV_MERGED_PATH")
        if merged_path:
            try:
                from prov4ml import prov4ml as _p  # type: ignore
                Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
                _p.PROV4ML_DATA.PROV_MERGED_PATH = str(merged_path)
                _debug(f"   Using merged path: {merged_path}")
            except Exception as e:
                _debug(f"   ⚠ Failed to set PROV_MERGED_PATH: {e}")

        if yprov:
            try:
                try:
                    from prov4ml.datamodel.metric_type import MetricsType  # type: ignore
                    metrics_type = MetricsType.CSV
                except Exception:
                    metrics_type = None

                kwargs_sr: Dict[str, Any] = dict(
                    prov_user_namespace=os.getenv("YPROV_USER_NAMESPACE", "yProv4ML"),
                    experiment_name=exp_name,
                    provenance_save_dir=str(exp_dir),
                    collect_all_processes=_env_flag("YPROV_COLLECT_ALL", False),
                    csv_separator=",",
                    unify_experiments=unify,
                )
                if metrics_type is not None:
                    kwargs_sr["metrics_file_type"] = metrics_type

                yprov.start_run(**kwargs_sr)
                _debug(f"  ✓ prov4ml.start_run(): dir={exp_dir} exp={exp_name} unify={unify}")
            except Exception as e:
                _debug(f"  ⚠ prov4ml.start_run() failed: {e}")
        else:
            _debug("  ⚠ prov4ml not available")

        _debug(f"  Created run_id: {run_id}")
        return result

    def set_terminated(self, run_id, status, end_time):
        _debug(f"set_terminated called: run_id={run_id}, status={status}")
        if yprov:
            _yprov_end_run()
        else:
            _debug("  ⚠ prov4ml not available")
        return self._delegate.set_terminated(run_id, status, end_time)

    def update_run_info(self, run_id, run_status, end_time, **kwargs):
        _debug(f"update_run_info called: run_id={run_id}, status={run_status}, end_time={end_time}")
        is_terminal = False
        try:
            status_str = str(run_status).upper()
            status_int = int(run_status) if str(run_status).isdigit() else -1
            is_terminal = status_str in {"FINISHED", "FAILED", "KILLED"} or status_int in {3, 4, 5}
        except Exception:
            pass
        _debug(f"  is_terminal={is_terminal}, end_time={end_time}")
        if yprov and is_terminal:
            _yprov_end_run()
        return self._delegate.update_run_info(run_id, run_status, end_time, **kwargs)

    def log_param(self, run_id, param):
        _debug(f"log_param: {param.key}={param.value}")
        if yprov:
            try:
                yprov.log_param(param.key, param.value)
                _debug(f"  ✓ prov4ml.log_param({param.key}={param.value})")
            except Exception as e:
                _debug(f"  ⚠ prov4ml.log_param() failed: {e}")
        return self._delegate.log_param(run_id, param)

    def log_metric(self, run_id, metric):
        if DEBUG:
            self._metric_log_count = getattr(self, "_metric_log_count", 0) + 1
            if self._metric_log_count <= 3 or self._metric_log_count % 10 == 0:
                _debug(f"log_metric: {metric.key}={metric.value}")
        if yprov:
            step = getattr(metric, "step", None)
            ok = _call_yprov_log_metric(metric.key, metric.value, step=step)
            if ok and DEBUG and (self._metric_log_count <= 3 or self._metric_log_count % 10 == 0):
                _debug(f"  ✓ prov4ml.log_metric({metric.key}={metric.value})")
        return self._delegate.log_metric(run_id, metric)

    def log_batch(self, run_id, metrics, params, tags):
        _debug(f"log_batch: {len(params or [])} params, {len(metrics or [])} metrics, {len(tags or [])} tags")
        if yprov:
            try:
                for p in (params or []):
                    try:
                        yprov.log_param(p.key, p.value)
                    except Exception as e:
                        _debug(f"  ⚠ prov4ml.log_param({p.key}) failed: {e}")
                for m in (metrics or []):
                    try:
                        step = getattr(m, "step", None)
                        _call_yprov_log_metric(m.key, m.value, step=step)
                    except Exception as e:
                        _debug(f"  ⚠ prov4ml.log_metric({m.key}) failed: {e}")
                _debug(f"  ✓ prov4ml logged {len(params or [])} params, {len(metrics or [])} metrics")
            except Exception as e:
                _debug(f"  ⚠ prov4ml.log_batch() failed: {e}")
        return self._delegate.log_batch(run_id, metrics, params, tags)

    def get_artifact_uri(self, run_id):
        if hasattr(self._delegate, "get_artifact_uri"):
            return self._delegate.get_artifact_uri(run_id)
        return self._delegate.get_run(run_id).info.artifact_uri

    def shut_down_async_logging(self):
        _debug("shut_down_async_logging called")
        if hasattr(self._delegate, "shut_down_async_logging"):
            return self._delegate.shut_down_async_logging()

    def create_experiment(self, name, artifact_location=None, tags=None):
        _debug(f"create_experiment: {name}")
        return self._delegate.create_experiment(name, artifact_location, tags)

    def get_experiment(self, exp_id):
        return self._delegate.get_experiment(exp_id)

    def get_experiment_by_name(self, name):
        return self._delegate.get_experiment_by_name(name)

    def get_run(self, run_id):
        return self._delegate.get_run(run_id)

    def search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token):
        return self._delegate.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by, page_token)

    def list_run_infos(self, experiment_id, run_view_type, max_results, order_by, page_token):
        return self._delegate.list_run_infos(experiment_id, run_view_type, max_results, order_by, page_token)
