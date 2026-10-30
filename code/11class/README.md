# NILM 11-Class Clean Pipeline

This is the corrected, independent project for the first five stages of the NILM experiment: classification of 11 appliance categories.

It is intentionally separate from the 26-state ON/OFF project.

Evidence:

- `docs/legacy-leakage-audit.md`
- `docs/evidence/legacy-title-band.png`
- `docs/evidence/pure-11class-example.png`

## Legacy failures corrected here

- Legacy images contained the appliance name in a rendered title.
- Legacy training read the entire PNG, so title pixels entered the CNN.
- Legacy training randomly split generated images without separating source recordings.
- Legacy cycle PKLs did not retain the originating CSV sample.
- Legacy training duplicated images when fewer than 55,000 existed.
- The legacy summation transform was called GADF even though it computes GASF.

## Corrected pipeline

1. Read `Save1_fixed/new_csv_fixed.pkl` metadata.
2. Read each source CSV and extract voltage-aligned 500-point cycles.
3. Uniformly reservoir-sample cycles per class.
4. Retain `source_id = csv_file|sample_original_index`.
5. Save pure RGB arrays with Pillow, without titles or Matplotlib margins.
6. Split by source sample, not by generated image.
7. Train the compact CNN using the grouped manifest.

The standard 11-class runner is fixed to **Gramian Angular Summation Field
(GASF)**. It does not expose a GADF switch in the main pipeline.

## Run in VS Code

Open:

```text
D:\论文1\nilm-11class-clean
```

Select the existing Python environment:

```text
D:\PLAID-projiect  2025.9.9\GLM-1\.venv\Scripts\python.exe
```

Run the smoke test:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -Smoke
```

Run the full experiment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1
```

Full output defaults to:

```text
D:\论文1\clean_data\plaid_11class_gasf_64
```

See `RUN_IN_VSCODE.md` for the complete workflow.
