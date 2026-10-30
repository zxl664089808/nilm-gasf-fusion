# 08 Device-level validation

For the 11-class project, the script automatically:

1. reads PLAID metadata;
2. builds a source-to-device map;
3. creates a leave-device-out split;
4. trains on the device-level split.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Dataset 11class
```

The 11-class device-level run defaults to the selected main representation I.
To run another representation explicitly:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run.ps1 `
  -Dataset 11class `
  -Channels "vi"
```

The 26-state project supports device-level training only after a
`source_id,device_id` map and `device_split.csv` are supplied.

This folder does not provide a second external dataset. Cross-dataset validation
requires additional data.

## Device experiment matrix

Run only split seed 42 first:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_matrix.ps1 `
  -SplitSeeds "42" `
  -Channels "i,vip"
```

For each channel this runs:

- image-only model;
- image + physical-feature model;
- image- and device-weighted evaluation.

After reviewing seed 42, repeat with:

```powershell
-SplitSeeds "43,44"
```
