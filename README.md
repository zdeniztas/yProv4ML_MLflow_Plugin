# yProv4ML_MLflow_Plugin
```
yProv4ML_MLflow_Plugin/
├── yprov_mlflow_plugin/       # Plugin source code
│   ├── tracking.py             # TrackingStore implementation
│   ├── artifacts.py            # ArtifactRepository implementation
│   └── prov_export.py          # PROV document generator
├── examples/                   # Example training scripts
│   ├── demo.py                 # CIFAR-10 example
│   ├── mnist_mlflow_demo.py    # MNIST with schedulers
│   ├── run_batch.py            # Grid/random search
│   ├── prov_reuse_plotting.py  # Provenance-driven reusable plotting
│   └── schemas/                # YAML provenance schemas
│       └── ml_pipeline.yaml    # Example ML pipeline schema
├── src/                        # Utilities
│   ├── prov_to_csv.py          # Convert PROV to CSV
│   ├── prov_join.py            # Join PROV JSON files
│   ├── prov_multiprocess.py    # Multi-process provenance handling
│   ├── prov_from_yaml.py       # PROV from YAML schemas
│   ├── prov_ro_crate.py        # RO-Crate packaging
│   ├── import_prov_to_mlflow.py # Import PROV to MLflow
│   └── decision_app.py         # Streamlit dashboard
├── test/                       # Test suite
└── pyproject.toml              # Plugin entry points

```

# yProv4ML MLflow Plugin

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.3+-green.svg)](https://mlflow.org/)

A powerful MLflow plugin that seamlessly integrates [W3C PROV](https://www.w3.org/TR/prov-overview/)-compliant provenance tracking with sustainability metrics for machine learning experiments built on the [yProv4ML (prov4ml)](https://github.com/zdeniztas/yProvML) 

## 🎯 Key Features

- **🔄 Zero-Code Integration**: Drop-in replacement for standard MLflow tracking
- **📊 W3C PROV Compliance**: Automatic generation of standardized provenance graphs
- **🌱 Sustainability Tracking**: Built-in CodeCarbon integration for carbon footprint monitoring
- **📈 Full MLflow Compatibility**: Works with existing MLflow workflows, UI, and tools
- **🔍 Comprehensive Metadata**: Captures hyperparameters, metrics, artifacts, and system info
- **📁 Dual Storage**: Maintains both MLflow runs and PROV JSON documents
- **🎨 Decision Support Dashboard**: Interactive Streamlit app for intelligent experiment recommendations



## 🏗️ Architecture Overview

The plugin implements two main MLflow extension points:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your ML Training Script                   │
│                  (uses standard MLflow API)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              yProv4ML MLflow Plugin Layer                    │
│  ┌──────────────────────┐    ┌─────────────────────────┐   │
│  │  YProvTrackingStore  │    │  YProvArtifactRepo      │   │
│  │  (tracking.py)       │    │  (artifacts.py)         │   │
│  └──────────┬───────────┘    └───────────┬─────────────┘   │
│             │                             │                  │
│             ▼                             ▼                  │
│  ┌──────────────────────┐    ┌─────────────────────────┐   │
│  │  Standard MLflow     │    │  Standard MLflow        │   │
│  │  FileStore/RestStore │    │  LocalArtifactRepo      │   │
│  └──────────┬───────────┘    └───────────┬─────────────┘   │
└─────────────┼─────────────────────────────┼─────────────────┘
              │                             │
              ▼                             ▼
     ┌────────────────┐            ┌──────────────────┐
     │  mlruns/       │            │  mlartifacts/    │
     │  (MLflow data) │            │  (artifacts)     │
     └────────────────┘            └──────────────────┘
              │                             │
              └──────────┬──────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  prov4ml Integration │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  data/prov/          │
              │  (PROV JSON files)   │
              └──────────────────────┘
```

### Component Breakdown

1. **YProvTrackingStore** (`tracking.py`)
   - Intercepts MLflow tracking calls (params, metrics, runs)
   - Delegates to standard MLflow storage (FileStore or RestStore)
   - Simultaneously logs to prov4ml for W3C PROV generation
   - Generates PROV JSON on run completion

2. **YProvArtifactRepo** (`artifacts.py`)
   - Handles artifact logging (models, configs, plots)
   - Mirrors artifacts to both MLflow and PROV storage
   - Maintains artifact provenance metadata

3. **PROV Export** (`prov_export.py`)
   - Converts MLflow run data to W3C PROV format
   - Extracts metadata from activities and entities
   - Generates JSON-LD compatible provenance documents

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for installing from source)

### Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/yProv4ML_MLflow_Plugin.git
cd yProv4ML_MLflow_Plugin

# Install the plugin in development mode
pip install -e .
```

#### **Install Requirements** 
```bash
pip install -r requirements.txt
```


### Verify Installation

```bash
# Check that the plugin is registered
python -c "import mlflow; print(mlflow.__version__)"

# Verify prov4ml is installed
python -c "import prov4ml; print('prov4ml installed successfully')"

# Check plugin entry points
python -c "from importlib.metadata import entry_points; eps = [ep for ep in entry_points().get('mlflow.tracking_store', []) if 'yprov' in ep.name]; print(f'Found {len(eps)} yprov entry points')"

# Test basic functionality
python -c "import mlflow; mlflow.set_tracking_uri('yprov+file:///tmp/test'); print('✅ Plugin activated successfully')"
```

### Platform-Specific Notes

#### **Windows**
```bash
# Use forward slashes or Path objects
mlflow.set_tracking_uri("yprov+file:///C:/Users/user/mlruns")

# Or use pathlib
from pathlib import Path
mlflow.set_tracking_uri(f"yprov+file:///{Path('C:/Users/user/mlruns').as_posix()}")
```

#### **WSL / Linux**
```bash
# Standard Unix paths work directly
mlflow.set_tracking_uri("yprov+file:///home/user/mlruns")
```

#### **macOS**
```bash
# Standard Unix paths
mlflow.set_tracking_uri("yprov+file:///Users/user/mlruns")
```

## ⚡ Quick Start

The plugin activates automatically when you use the `yprov+` URI scheme:

```python
import mlflow

# 🔑 CRITICAL: Use yprov+ prefix to activate the plugin
mlflow.set_tracking_uri("yprov+file:///path/to/mlruns")

# Standard MLflow code - no changes needed!
mlflow.set_experiment("my_experiment")

with mlflow.start_run(run_name="example_run"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    
    # Log metrics
    for epoch in range(5):
        mlflow.log_metric("loss", 0.5 - epoch * 0.05, step=epoch)
        mlflow.log_metric("accuracy", 0.7 + epoch * 0.05, step=epoch)
    
    # Log artifacts
    mlflow.log_artifact("model.pt", artifact_path="models")

# ✅ PROV JSON automatically generated in data/prov/my_experiment/
```


## 🔧 How It Works

### Plugin Activation

The plugin registers itself via setuptools entry points defined in `pyproject.toml`:

```toml
[project.entry-points."mlflow.tracking_store"]
"yprov+http" = "yprov_mlflow_plugin.tracking:YProvTrackingStore"
"yprov+file" = "yprov_mlflow_plugin.tracking:YProvTrackingStore"

[project.entry-points."mlflow.artifact_repository"]
"yprov+file" = "yprov_mlflow_plugin.artifacts:YProvArtifactRepo"
"yprov+http" = "yprov_mlflow_plugin.artifacts:YProvArtifactRepo"
```

When MLflow sees a URI starting with `yprov+`, it automatically routes calls through the plugin.

### Data Flow

#### 1. Run Creation (`mlflow.start_run()`)

```python
# User code
with mlflow.start_run(run_name="my_run"):
    # ...

# What happens internally:
# 1. YProvTrackingStore.create_run() is called
# 2. Creates standard MLflow run in mlruns/
# 3. Calls prov4ml.start_run() with:
#    - experiment_name
#    - provenance_save_dir (data/prov/experiment_name/)
#    - metrics_file_type (CSV or NetCDF)
# 4. Returns MLflow Run object
```

#### 2. Parameter Logging (`mlflow.log_param()`)

```python
# User code
mlflow.log_param("learning_rate", 0.001)

# What happens internally:
# 1. YProvTrackingStore.log_param() is called
# 2. Logs to MLflow: mlruns/exp_id/run_id/params/learning_rate
# 3. Calls prov4ml.log_param("learning_rate", 0.001)
# 4. Stores in PROV context for later JSON generation
```

#### 3. Metric Logging (`mlflow.log_metric()`)

```python
# User code
mlflow.log_metric("loss", 0.5, step=1)

# What happens internally:
# 1. YProvTrackingStore.log_metric() is called
# 2. Logs to MLflow: mlruns/exp_id/run_id/metrics/loss
# 3. Calls prov4ml.log_metric("loss", 0.5, step=1)
# 4. Writes to CSV/NetCDF: data/prov/experiment/metrics/loss.csv
```

#### 4. Artifact Logging (`mlflow.log_artifact()`)

```python
# User code
mlflow.log_artifact("model.pt", artifact_path="models")

# What happens internally:
# 1. YProvArtifactRepo.log_artifact() is called
# 2. Copies to MLflow: mlartifacts/run_id/models/model.pt
# 3. Calls prov4ml.log_artifact() to record in PROV
# 4. Stores artifact metadata for provenance graph
```

#### 5. Run Completion (`mlflow.end_run()` or context exit)

```python
# User code (explicit or implicit)
mlflow.end_run()
# or
# with mlflow.start_run():
#     ...  # auto-closes on exit

# What happens internally:
# 1. YProvTrackingStore.set_terminated() is called
# 2. Calls prov4ml.end_run(create_graph=True)
# 3. Generates PROV JSON in data/prov/experiment/prov_*.json
# 4. Updates MLflow run status


## 📚 Usage Examples

### Example 1: Simple Classification

```python
import mlflow
from pathlib import Path

# Setup
mlflow.set_tracking_uri("yprov+file:///home/user/mlruns")
mlflow.set_experiment("iris_classification")

with mlflow.start_run(run_name="logistic_regression"):
    # Parameters
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 100)
    
    # Training (your code here)
    # ...
    
    # Metrics
    mlflow.log_metric("train_accuracy", 0.95)
    mlflow.log_metric("test_accuracy", 0.93)
    mlflow.log_metric("f1_score", 0.94)
    
    # Save model
    mlflow.log_artifact("model.pkl", artifact_path="models")

# PROV JSON available at: data/prov/iris_classification/prov_*.json
```

### Example: Batch Experiments

Use the included `run_batch.py` script for systematic hyperparameter sweeps:

```bash
# Grid search
python examples/run_batch.py \
    --script examples/demo.py \
    --mode grid \
    --grid batch_size=8,16,32,64 \
    --grid lr=1e-3,1e-4,1e-5 \
    --grid epochs=5,10 \
    --jobs 4

# Random search
python examples/run_batch.py \
    --script examples/demo.py \
    --mode random \
    --n 50 \
    --rand batch_size=choice(8,16,32,64,128) \
    --rand lr=log10(-5,-2) \
    --rand epochs=int(5,20) \
    --jobs 4
```

## 🎨 Dashboard Usage

### Decision Support Dashboard

The plugin includes an interactive Streamlit dashboard for analyzing experiments and getting intelligent recommendations:

```bash
# Start the dashboard
streamlit run src/decision_app.py
```

#### Dashboard Features

1. **📊 Overview Metrics**
   - Total experiments tracked
   - Parameter detection
   - Mean performance and sustainability metrics

2. **🔬 Solution Space Analysis**
   - **Pairwise Analysis**: 2D slices of multi-dimensional hyperparameter space
   - **Heatmaps**: Performance across parameter combinations
   - **Main Effects**: Individual parameter impact
   - **Correlations**: Relationships between parameters and metrics

3. **🎯 Pareto Frontier**
   - Visualization of trade-offs between accuracy and sustainability
   - Interactive plots with hover tooltips
   - Identify optimal configurations

4. **💡 Intelligent Recommendations**
   - **EXPLOIT**: Refine around best known configurations
   - **EXPLORE**: Test underexplored parameter regions
   - **BALANCE**: Optimize accuracy vs. sustainability trade-off
   - **INTERPOLATE**: Fill gaps in tested parameter values

5. **🎲 Advanced Clustering with SHAP**
   - Automatic experiment grouping (KMeans, DBSCAN, etc.)
   - SHAP feature importance for cluster assignments
   - Understand which parameters drive experiment similarity

#### Using the Dashboard

1. **Export Experiments to CSV**:
```bash
# Convert PROV JSONs to CSV
python src/prov_to_csv.py --root data/prov --out-dir data/unified
```

2. **Launch Dashboard**:
```bash
streamlit run decision_app_updated.py
```

3. **Upload CSV** via the sidebar

4. **Select Metrics**:
   - Choose performance metric (e.g., "ACC_val")
   - Choose sustainability metric (e.g., "emissions (tCO2eq)")

5. **Explore Visualizations**:
   - Navigate through tabs for different analyses
   - Adjust clustering parameters in the sidebar
   - Generate and download recommendations

## 🛠️ Utilities

### 1. PROV to CSV Converter

Convert PROV JSON files to structured CSV format for analysis:

```bash
# Basic usage
python src/prov_to_csv.py \
    --root data/prov \
    --out-dir data/unified

# With debug output
python src/prov_to_csv.py \
    --root data/prov \
    --out-dir data/unified \
    --debug
```

**Features**:
- Automatically detects and reads both CSV and NetCDF metric files
- Extracts parameters from PROV activities
- Extracts metrics from entity files
- Splits output by experiment for easy analysis
- Removes empty columns per experiment

**Output Structure**:
```
data/unified/
├── experiment1.csv    # All runs for experiment1
├── experiment2.csv    # All runs for experiment2
└── experiment3.csv    # All runs for experiment3
```

### 2. Import PROV to MLflow

Import existing PROV provenance data back into MLflow:

```bash
python src/import_prov_to_mlflow.py \
    --prov-root data/prov/usecase \
    --experiment-name imported_experiments
```

**Use Cases**:
- Migrate PROV data to MLflow
- Share experiments with collaborators who use MLflow
- Visualize PROV data in MLflow UI

### 3. PROV JSON Joining

Join multiple PROV JSON files into a unified provenance document, de-duplicating
entities and activities by their identifiers. Based on initial work by Deniz and
the [y2Graph](https://github.com/HPCI-Lab/y2Graph) library for joining JSONs by IDs.

```bash
# Join all prov JSONs under a directory
python src/prov_join.py --root data/prov/my_experiment --out merged_prov.json

# Join specific files
python src/prov_join.py --files run1/prov_0.json run2/prov_0.json --out merged.json

# Join with source traceability tags
python src/prov_join.py --root data/prov --out merged.json --tag-sources

# Join multi-process provenance (GR0/GR1 kept separate)
python src/prov_join.py --root data/prov --out merged.json --multi-process
```

**Features**:
- De-duplicates entities and activities by stable IDs (label, path, run_id)
- Preserves all relation types (used, wasGeneratedBy, wasDerivedFrom, etc.)
- Optional `--tag-sources` to track which file each element came from
- Multi-process mode keeps per-rank metrics separate while merging shared entities

### 4. Multi-Process Provenance

Handle distributed training provenance where prov4ml generates separate metrics
per process rank (metrics_GR0/, metrics_GR1/, ...).

```bash
# Join multi-process provenance for a single run
python src/prov_multiprocess.py --run-dir data/prov/experiment/run_0 --out merged/

# Process all runs with aggregate statistics
python src/prov_multiprocess.py --experiment-dir data/prov/my_experiment --out merged/ --stats

# Full workflow: join + statistics + CSV export
python src/prov_multiprocess.py --experiment-dir data/prov/my_experiment --out merged/ --stats --csv
```

**Features**:
- Auto-discovers metrics_GR0/, metrics_GR1/, etc. directories
- Computes aggregate statistics (mean, std, min, max) across processes
- Joins PROV documents without duplicating shared entities (model, dataset, config)
- Exports per-experiment CSV summary of aggregated metrics

### 5. YAML-Based Provenance Schemas

Create PROV JSON documents and visualizations from simple YAML files,
useful for quick schematization (e.g., paper diagrams) without running experiments.

```bash
# Generate PROV JSON from a YAML schema
python src/prov_from_yaml.py --schema examples/schemas/ml_pipeline.yaml --out output/pipeline.json

# Also generate DOT graph and SVG image
python src/prov_from_yaml.py --schema examples/schemas/ml_pipeline.yaml \
    --out output/pipeline.json --dot --img
```

See `examples/schemas/ml_pipeline.yaml` for a complete example schema.

### 6. RO-Crate Packaging

Package experiment code, provenance, and artifacts into
[RO-Crate](https://www.researchobject.org/ro-crate/) archives. RO-Crates are
snapshots of the current codebase, very useful for reproducibility and for
reusing the exact code that produced specific results (e.g., for plotting).

```bash
# Package a single run
python src/prov_ro_crate.py --run-dir data/prov/experiment/run_0 --out crates/

# Package with source code included
python src/prov_ro_crate.py --run-dir data/prov/experiment/run_0 \
    --include-src examples/demo.py yprov_mlflow_plugin/ \
    --out crates/

# Package with git commit info for traceability
python src/prov_ro_crate.py --run-dir data/prov/experiment/run_0 --out crates/ --git-info

# Package all runs in an experiment
python src/prov_ro_crate.py --experiment-dir data/prov/my_experiment --out crates/
```

**Features**:
- Creates RO-Crate 1.1 compliant ZIP archives
- Includes ro-crate-metadata.json with full provenance context
- Optionally bundles source code for full reproducibility
- Captures git commit info for traceability

### 7. Provenance-Driven Reusable Plotting

The same plotting code works for ANY experiment that produced PROV JSON,
demonstrating the power of provenance for code reuse across use cases.

```bash
# Plot metrics from any experiment's PROV data
python examples/prov_reuse_plotting.py --prov-dir data/prov/my_experiment

# Compare CIFAR vs MNIST experiments side by side
python examples/prov_reuse_plotting.py \
    --prov-dir data/prov/cifar_exp data/prov/mnist_exp \
    --compare

# Plot from an RO-Crate archive
python examples/prov_reuse_plotting.py --ro-crate crates/run_0_ro-crate.zip

# Save plots to a directory
python examples/prov_reuse_plotting.py --prov-dir data/prov/my_experiment --out plots/
```

**Key Insight**: Because provenance provides a uniform data interface (metrics,
parameters, artifacts), the same analysis/plotting code can be applied to any
experiment regardless of the underlying ML task (classification, regression,
generative, etc.).

## 🔍 Configuration

### Environment Variables

Control plugin behavior via environment variables:

```bash
# PROV output directory (base path)
export YPROV_OUT_DIR="/path/to/prov/output"
# Default: data/prov

# Explicit full prov path (takes precedence over YPROV_OUT_DIR)
export YPROV_PROV_PATH="/explicit/path/to/prov"

# Output format control: comma-separated list of formats
# Options: "json", "dot", "img", "svg", "ro-crate"
export YPROV_OUTPUT_FORMAT="json"
# Default: json
# Example: YPROV_OUTPUT_FORMAT="json,dot,img"  (generates all three)

# Enable debug logging
export YPROV_DEBUG="1"
# Default: 0

# User namespace for PROV documents
export YPROV_USER_NAMESPACE="MyOrganization"
# Default: yProv4ML

# Collect all system processes (more overhead)
export YPROV_COLLECT_ALL="1"
# Default: 0

# Unify experiments in single PROV document
export YPROV_UNIFY="1"
# Default: 0

# Custom merged PROV file path (when YPROV_UNIFY=1)
export YPROV_MERGED_PATH="/path/to/merged_prov.json"
# Default: data/prov/merged_provenance.json
```

### Python Configuration

```python
import os
import mlflow

# Set configuration before mlflow.set_tracking_uri()
os.environ["YPROV_OUT_DIR"] = "/custom/prov/dir"
os.environ["YPROV_DEBUG"] = "1"
os.environ["YPROV_USER_NAMESPACE"] = "MyProject"

# Then activate plugin
mlflow.set_tracking_uri("yprov+file:///path/to/mlruns")
```

### Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/yProv4ML_MLflow_Plugin.git
cd yProv4ML_MLflow_Plugin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest test/ -v
```


### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Areas for Contribution

- 🐛 **Bug fixes**: See [Issues](https://github.com/yourusername/yProv4ML_MLflow_Plugin/issues)
- 📚 **Documentation**: Improve guides, add examples
- ✨ **Features**: 
  - Additional dashboard visualizations
  - More export formats (CSV, Parquet, etc.)
  - Integration with other tracking tools
  - Performance optimizations
- 🧪 **Testing**: Increase test coverage, add edge cases


## 🙏 Acknowledgments

- [MLflow](https://mlflow.org/) - Open source ML lifecycle platform
- [yProv4ML (prov4ml)](https://github.com/zdeniztas/yProvML) - W3C PROV provenance tracking
- [CodeCarbon](https://codecarbon.io/) - Carbon emissions tracking
- [W3C PROV](https://www.w3.org/TR/prov-overview/) - Provenance standard


---
