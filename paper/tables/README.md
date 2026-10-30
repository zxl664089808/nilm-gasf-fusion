# Frozen result tables

The CSV files in this directory are snapshots of the audited results used by
the v0.3 manuscript.

- `11class_recording_*`: unseen-recording channel comparison.
- `11class_device_*`: leave-device-out results and run-level records.
- `26state_image_*`: image-only channel and seed results.
- `26state_fusion_runs.csv`: three physical-fusion seeds.
- `26state_model_comparison.csv`: fusion and multitask comparison.
- `leakage_audit.csv`: reduced-scale title-band diagnostics.

The manuscript summarizes three training seeds under one fixed data split.
Standard deviations therefore describe optimization variability, not
independent resampling uncertainty.

Exploratory raw-waveform and priority-baseline outputs are not included here.
They remain outside the unified repository until their code and result
provenance are reviewed and committed.
