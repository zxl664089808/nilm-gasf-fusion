# Recommended execution order

All experiments must use the corrected grouped-split manifests. Do not mix
results from smoke runs or the old image-level-split experiments.

## Phase 0: preflight

1. Confirm the working tree and record the commit hashes:

```powershell
git -C D:\论文1\nilm-clean-pipeline log -1 --oneline
git -C D:\论文1\nilm-11class-clean log -1 --oneline
git -C D:\论文1\nilm-validation-suite log -1 --oneline
```

2. Record SHA256 hashes for the two grouped-split manifests:

```powershell
Get-FileHash "D:\论文1\clean_data\plaid_26state_gasf_64\grouped_split.csv" -Algorithm SHA256
Get-FileHash "D:\论文1\clean_data\plaid_11class_gasf_64\grouped_split.csv" -Algorithm SHA256
```

3. Confirm both image caches exist:

```text
D:\论文1\clean_data\plaid_26state_gasf_64\images_64.npy
D:\论文1\clean_data\plaid_11class_gasf_64\images_64.npy
```

4. Do not change the manifest, split seed, image cache, or evaluation code
between variants of the same comparison.

## Phase 1: channel ablation

Run the 26-state experiment first because it is the hardest task.

```powershell
cd D:\论文1\nilm-validation-suite
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\01_channel_ablation\run.ps1 -Dataset 26state
```

Then run the 11-class experiment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\01_channel_ablation\run.ps1 -Dataset 11class
```

Review:

```text
01_channel_ablation\runs\26state\benchmark_summary.csv
01_channel_ablation\runs\11class\benchmark_summary.csv
```

Decision gate:

- If P alone remains strongest for 26-state, do not assume RGB fusion is better.
- If V+I+P is clearly strongest for 11-class, retain it as the main representation.
- Do not tune on the test set. Use validation macro-F1 for selection.

## Phase 2: encoding baseline

Run the raw 1D and V-I trajectory baselines for the 26-state task:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\02_encoding_ablation\run.ps1
```

Outputs:

```text
02_encoding_ablation\runs\raw1d\metrics.json
02_encoding_ablation\runs\vi_trajectory\metrics.json
```

Decision gate:

- Compare against the corrected GASF baseline.
- If raw 1D matches or exceeds GASF with substantially lower cost, the paper
  must explain why the image representation is still justified.
- If GASF is better, use the result as evidence for the representation.

## Phase 3: true resolution ablation

This phase requires new 96 and 128 datasets. Do not simply upscale the 64×64
files.

Generate the 96 and 128 data according to
`../nilm-clean-pipeline/EXPERIMENTS.md`, create grouped splits, and build image
caches for each resolution.

Then run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\03_resolution_ablation\run.ps1
```

Decision gate:

- Move to the highest resolution only if the validation macro-F1 gain justifies
  the parameter, memory, and inference cost.

## Phase 4: model comparison

Run after the representation and resolution choices are reasonable.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\04_model_comparison\run.ps1 -Dataset 26state
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\04_model_comparison\run.ps1 -Dataset 11class
```

Compare:

- parameter count;
- validation macro-F1;
- test macro-F1;
- training time;
- inference memory and latency if measured.

Decision gate:

- Keep the 4-layer CNN if a larger model does not provide a meaningful,
  repeatable improvement.
- Report both accuracy and efficiency; do not select only by the highest test
  accuracy.

## Phase 5: physical features and multitask model

These are the most important targeted improvements for the hard classes.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\05_physics_fusion\run.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\06_multitask_type_state\run.ps1
```

Inspect whether they improve:

- Coffee maker_OFF recall;
- Water kettle_ON recall;
- Fridge_OFF recall;
- macro-F1, not only overall accuracy.

If either model improves macro-F1 while preserving the grouped split, it
becomes the candidate for the final paper model.

## Phase 6: seed robustness and device validation

Run seeds only for the selected final model configurations:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\07_seed_robustness\run.ps1 -Dataset 26state
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\07_seed_robustness\run.ps1 -Dataset 11class
```

Run leave-device-out validation for the 11-class project:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\08_device_validation\run.ps1 -Dataset 11class
```

Decision gate:

- Report mean and standard deviation across seeds.
- Report unseen-recording and leave-device-out results separately.
- Do not call leave-device-out a second-dataset result.

## Phase 7: cross-dataset validation

A second dataset is not present in the current workspace. This phase remains
blocked until a dataset such as WHITED, UK-DALE, or another labeled high-frequency
dataset is supplied with a compatible manifest.

## What to put in the paper

At minimum:

1. Corrected baseline result.
2. Channel ablation.
3. GASF vs raw 1D vs V-I.
4. Resolution selection.
5. Model and efficiency comparison.
6. Physics fusion or multitask improvement.
7. Multi-seed mean ± SD.
8. Unseen-recording and leave-device-out discussion.

The quick screening results in `../00_project_admin/improvement_progress.md`
are not final paper numbers.
