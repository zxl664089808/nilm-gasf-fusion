# NILM Validation Suite

This project separates the eight validation studies into independent experiment
folders while reusing the corrected implementations in:

- `../nilm-clean-pipeline`: 26-state experiments
- `../nilm-11class-clean`: 11-class experiments

Each experiment has:

- its own `README.md`;
- its own `run.ps1`;
- its own `runs/` output directory.

Run scripts require PowerShell execution-policy bypass:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\01_channel_ablation\run.ps1
```

## Experiments

| Folder | Validation | Main data |
|---|---|---|
| `01_channel_ablation` | V, I, P, V+I, V+I+P | 11-class and 26-state |
| `02_encoding_ablation` | GASF vs raw 1D vs V-I trajectory | 26-state |
| `03_resolution_ablation` | 64, 96, 128 input | 11-class and 26-state |
| `04_model_comparison` | Simple CNN, ResNet-18, EfficientNet-B0 | 11-class and 26-state |
| `05_physics_fusion` | GASF + physical-feature MLP | 26-state |
| `06_multitask_type_state` | Shared encoder with type/state heads | 26-state |
| `07_seed_robustness` | Seeds 42, 43, 44 | 11-class and 26-state |
| `08_device_validation` | Leave-device-out split | 11-class; generic for 26-state |

The shared implementations live in the two corrected code repositories. Each
validation folder defines its own experiment configuration and output location.
