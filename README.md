# NILM GASF Fusion

Unified reproducibility repository for the paper draft:

**Leakage-Aware Multi-Physical GASF Imaging for Plug-Load Appliance and State Recognition**

## Current manuscript

- Source: `paper/manuscript/manuscript_electronics_v0.3.md`
- Numbered citations: `paper/manuscript/manuscript_electronics_v0.3_numbered.md`
- Browser preview: `paper/manuscript/manuscript_electronics_v0.3_preview.html`
- Cover letter: `paper/manuscript/electronics_cover_letter.md`
- Submission status: `paper/submission/pre_submission_status_v0.3.md`

Rebuild the numbered manuscript and HTML preview with:

```powershell
D:\Anaconda\python.exe paper\manuscript\build_preview.py
```

Version 0.2 and its preview are retained as development history.

## Structure

```text
code/26state/     Corrected 26-state implementation
code/11class/     Corrected 11-class implementation
code/validation/  Independent validation and ablation runners
paper/manuscript/ English manuscript and preview
paper/figures/    Paper figures and captions
paper/tables/     Generated result tables
paper/references/ Verified BibTeX subset
paper/submission/ Submission preparation
paper/project_admin/ Audits and experiment records
```

## Final model choices

### 11-class appliance recognition

- Main model: three-channel GASF image CNN.
- Unseen-recording: 92.86 +/- 0.13% accuracy and 92.82 +/- 0.09% macro-F1.
- Unseen-device: 80.90 +/- 6.61% image accuracy and 90.62 +/- 2.76% mean
  device accuracy across three device splits.
- Single-current I is the lightweight recording-level baseline.

### 26-state appliance and ON/OFF recognition

- Main model: three-channel GASF image CNN plus 14 physical descriptors.
- Test accuracy: 96.23 +/- 0.23%.
- Test macro-F1: 96.23 +/- 0.24%.
- Parameters: 117,498.
- The physical branch stabilizes Coffee maker and Water kettle recognition.

## Data

The raw PLAID data, generated images, image caches, trained model weights, and
run directories are intentionally excluded from Git. See
`paper/manuscript/manuscript_electronics_v0.2.md` for the dataset and
preprocessing description.

## Reproduction

Each code subproject contains its own README:

- `code/11class/README.md`
- `code/26state/README.md`
- `code/validation/README.md`

The validation project documents the ordered experiment workflow in
`code/validation/RUN_ORDER.md`.

## Provenance

The initial unified snapshot was assembled from committed source snapshots of
three independent working repositories. See `PROVENANCE.md`.
