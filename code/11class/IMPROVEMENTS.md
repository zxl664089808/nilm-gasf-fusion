# Improvement experiments for the 11-class project

The existing full run remains the baseline.

## Channel ablation

```powershell
python .\scripts\run_benchmark.py `
  --manifest "D:\论文1\clean_data\plaid_11class_gasf_64\grouped_split.csv" `
  --train-script .\scripts\train_classifier.py `
  --output-root "D:\论文1\clean_data\experiments\11class_channels" `
  --only channels
```

## Model comparison

```powershell
python .\scripts\run_benchmark.py `
  --manifest "D:\论文1\clean_data\plaid_11class_gasf_64\grouped_split.csv" `
  --train-script .\scripts\train_classifier.py `
  --output-root "D:\论文1\clean_data\experiments\11class_models" `
  --only models
```

## Multiple seeds

```powershell
python .\scripts\run_benchmark.py `
  --manifest "D:\论文1\clean_data\plaid_11class_gasf_64\grouped_split.csv" `
  --train-script .\scripts\train_classifier.py `
  --output-root "D:\论文1\clean_data\experiments\11class_seeds" `
  --only seeds `
  --seeds 42,43,44
```

## Device-level validation

Build the PLAID metadata map:

```powershell
python .\scripts\build_device_map.py `
  --metadata-docx "D:\PLAID-projiect  2025.9.9\sub_meta.docx" `
  --manifest "D:\论文1\clean_data\plaid_11class_gasf_64\grouped_split.csv" `
  --output "D:\论文1\clean_data\plaid_11class_gasf_64\device_map.csv"
```

Create a leave-device-out split:

```powershell
python .\scripts\create_device_split.py `
  --manifest "D:\论文1\clean_data\plaid_11class_gasf_64\dataset_manifest.csv" `
  --device-map "D:\论文1\clean_data\plaid_11class_gasf_64\device_map.csv" `
  --output "D:\论文1\clean_data\plaid_11class_gasf_64\device_split.csv"
```

Train with:

```powershell
python .\scripts\train_classifier.py `
  --manifest "D:\论文1\clean_data\plaid_11class_gasf_64\device_split.csv" `
  --output-dir "D:\论文1\clean_data\experiments\11class_device_split"
```

## True resolution ablation

Generate independent 96/128 datasets with `generate_dataset.py --image-size`,
then use `run_benchmark.py --manifest-96` and `--manifest-128`.
