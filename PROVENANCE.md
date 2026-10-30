# Source provenance

The unified repository contains clean, committed snapshots of three working
repositories. Uncommitted changes from other terminals were intentionally not
included.

## Included snapshots

| Component | Source commit |
|---|---|
| `code/26state` | `a34aa75a6624d82df3b116f8ac6b23687531e6f1` |
| `code/11class` | `1bd2a84499cc0d567dae25188cf6d20e0e4d4c35` |
| `code/validation` | `8d9bc9b02183564fc40706b1ba499fc5d5b1b11a` |

## Excluded working-tree changes

The following changes existed in other terminal sessions and were not part of
the committed source snapshots:

- additional matched raw-waveform and V-I capacity experiments in the
  26-state working tree;
- `09_priority_baselines` and related README/RUN_ORDER edits in the validation
  working tree.

They remain in their original local working directories and are not represented
as part of the submission pipeline in this snapshot.

## Paper snapshot

The included manuscript is the restored intended draft:

`paper/manuscript/manuscript_electronics_v0.2.md`

The unintended broad-comparison draft is retained only in the original local
working directory at `D:\论文1\06_manuscript\archive\`. It is not included in
this unified repository and is not the manuscript source used by
`paper/manuscript/build_preview.py`.
