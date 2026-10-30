# NILM Clean Pipeline

This repository is a corrected replacement for the legacy PLAID image pipeline. It removes the label leakage found in the original scripts and makes recording-level evaluation explicit.

## What was wrong

The legacy image generators called:

```python
plt.title(f"{appliance_name} {status} ...")
plt.savefig(...)
```

Training then read the whole PNG, so the rendered class name and ON/OFF state became model input features. The legacy training scripts also split generated images randomly instead of splitting by source recording.

## What this project fixes

- Saves RGB arrays directly with `PIL.Image.fromarray`; no title, axes, text, colorbar, or Matplotlib figure.
- Implements GASF and GADF separately with explicit method names.
- Preserves `source_id` for every generated image.
- Creates train/validation/test splits by source recording.
- Rejects the legacy 11-class cycle files if they lack source IDs.
- Removes the image-copy fallback from the 55,000-image training path.
- Adds tests for numerical transforms, pure image saving, and grouped splits.

Detailed legacy evidence:

- `docs/legacy-leakage-audit.md`
- `docs/evidence/legacy-title-band.png`
- `docs/evidence/pure-gadf-example.png`

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

The original data are expected to remain outside this repository.

For a guided VS Code workflow, see `RUN_IN_VSCODE.md`. On Windows, the quick
end-to-end smoke test is:

```powershell
.\scripts\run_pipeline.ps1 -Smoke
```

## Generate the 26-state dataset

```powershell
python scripts\generate_26state_dataset.py `
  --cycles-root "D:\PLAID-projiect  2025.9.9\GLM-1\extracted_cycles_output" `
  --output-root "D:\paper-data\plaid-26state-clean" `
  --max-cycles-per-label 5000 `
  --gramian gasf `
  --image-size 64
```

The output contains:

```text
plaid-26state-clean\
├── images\
│   └── <label>\
│       └── <source-record>__<cycle>.png
└── dataset_manifest.csv
```

## Create grouped splits

```powershell
python scripts\create_grouped_split.py `
  --manifest "D:\paper-data\plaid-26state-clean\dataset_manifest.csv" `
  --output "D:\paper-data\plaid-26state-clean\grouped_split.csv"
```

## Validate

```powershell
python scripts\validate_dataset.py `
  --manifest "D:\paper-data\plaid-26state-clean\grouped_split.csv"
```

Validation checks:

- every image exists;
- every source recording belongs to one split;
- a sample of images does not contain the historical white title band.

## Train

```powershell
python scripts\train_from_manifest.py `
  --manifest "D:\paper-data\plaid-26state-clean\grouped_split.csv" `
  --output-dir "D:\paper-runs\26state-clean"
```

The metrics JSON and confusion matrix are written to the output directory.

## 11-class legacy data

The historical 11-class cycle PKLs do not contain recording IDs. `generate_11class_dataset.py` deliberately refuses to invent them. Re-extract those cycles while retaining `source_ids`, or retain the 11-class experiment only as a non-publishable historical baseline.

## Terminology

- GASF: `sin(phi_i + phi_j)`
- GADF: `sin(phi_i - phi_j)`

The legacy scripts used the summation form while calling it GADF. The method name must match the implementation in the paper.
