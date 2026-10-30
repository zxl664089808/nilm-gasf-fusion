# Prioritized improvement experiments

The existing `runs/conservative_cnn` result remains the baseline.

## 1. Channel ablation

```powershell
python .\scripts\run_benchmark.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --train-script .\scripts\train_from_manifest.py `
  --output-root "D:\论文1\clean_data\experiments\26state_channels" `
  --only channels
```

Channels are `v`, `i`, `p`, `vi`, and `vip`.

## 2. Encoding ablation

Build a waveform cache first:

```powershell
python .\scripts\build_raw_cache.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --output "D:\论文1\clean_data\plaid_26state_gasf_64\raw_waveforms.npy"
```

Then compare:

```powershell
python .\scripts\train_raw.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --cache "D:\论文1\clean_data\plaid_26state_gasf_64\raw_waveforms.npy" `
  --output-dir "D:\论文1\clean_data\experiments\26state_raw1d" `
  --input-mode raw1d

python .\scripts\train_raw.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --cache "D:\论文1\clean_data\plaid_26state_gasf_64\raw_waveforms.npy" `
  --output-dir "D:\论文1\clean_data\experiments\26state_vi" `
  --input-mode vi
```

## 3. True resolution ablation

Generate independent 96 and 128 datasets:

```powershell
python .\scripts\generate_26state_dataset.py `
  --cycles-root "D:\PLAID-projiect  2025.9.9\GLM-1\extracted_cycles_output" `
  --output-root "D:\论文1\clean_data\plaid_26state_gasf_96" `
  --image-size 96

python .\scripts\generate_26state_dataset.py `
  --cycles-root "D:\PLAID-projiect  2025.9.9\GLM-1\extracted_cycles_output" `
  --output-root "D:\论文1\clean_data\plaid_26state_gasf_128" `
  --image-size 128
```

Create grouped splits for both, then pass them with `--manifest-96` and
`--manifest-128` to `run_benchmark.py`.

## 4. Model comparison

```powershell
python .\scripts\run_benchmark.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --train-script .\scripts\train_from_manifest.py `
  --output-root "D:\论文1\clean_data\experiments\26state_models" `
  --only models
```

Models: current 4-layer CNN, ResNet-18 from scratch, EfficientNet-B0 from
scratch.

## 5. Physics-feature fusion

```powershell
python .\scripts\build_physical_features.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --output "D:\论文1\clean_data\plaid_26state_gasf_64\physical_features.csv"

python .\scripts\train_hybrid.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --physical-features "D:\论文1\clean_data\plaid_26state_gasf_64\physical_features.csv" `
  --output-dir "D:\论文1\clean_data\experiments\26state_hybrid"
```

Features: RMS voltage/current, active/apparent power, power factor, peak,
crest factor, current THD, 3rd/5th/7th harmonic ratios, mean absolute
current/power, and log active power.

## 6. Multi-task type/state model

```powershell
python .\scripts\train_multitask.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --output-dir "D:\论文1\clean_data\experiments\26state_multitask"
```

The model has a shared CNN encoder and separate appliance-type and ON/OFF
heads. Joint and per-head accuracies are reported.

## 7. Multiple seeds

```powershell
python .\scripts\run_benchmark.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" `
  --train-script .\scripts\train_from_manifest.py `
  --output-root "D:\论文1\clean_data\experiments\26state_seeds" `
  --only seeds `
  --seeds 42,43,44
```

## 8. Device-level validation and second dataset

Device-level validation requires a CSV mapping:

```text
source_id,device_id
```

Then run:

```powershell
python .\scripts\create_device_split.py `
  --manifest "D:\论文1\clean_data\plaid_26state_gasf_64\dataset_manifest.csv" `
  --device-map "path\to\device_map.csv" `
  --output "D:\论文1\clean_data\plaid_26state_gasf_64\device_split.csv"
```

The 11-class project has an automatic PLAID metadata mapping tool. A second
dataset is not present in the current workspace and must be supplied before
cross-dataset validation can run.
