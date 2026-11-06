# Pre-submission status v0.3.2

Date: 2026-10-30

## Target

- Journal: *Electronics* (MDPI)
- Article type: research article
- Title: "Leakage-Aware Multi-Physical GASF Imaging for Plug-Load Appliance and
  State Recognition"
- Manuscript: `paper/manuscript/manuscript_electronics_v0.3.md`

## Completed checks

- The abstract is 208 words.
- The manuscript contains six figures and five captioned result tables.
- Twenty-four cited references resolve to entries in
  `paper/references/references_cited.bib`.
- No mojibake or rendered-label placeholders remain in the source manuscript.
- The six result and method figures were copied into `paper/figures` with
  vector PDF and high-resolution PNG versions.
- Frozen CSV summaries were copied into `paper/tables`.
- Recording-grouped and device-disjoint metrics are reported separately.
- The I-channel device result is identified as a single split, while V+I+P uses
  three splits.
- The 100% heating-load recall is limited to the fixed held-out split and is not
  presented as universal generalization.

## Remaining author inputs

1. The corresponding email address for Qinghua Zhou.
2. Confirmation of whether the funding statement should remain "no external
   funding," because Qinghua Zhou is assigned the Funding acquisition role.
3. A public DOI-bearing archive URL, preferably a tagged Zenodo release.

## Technical work before submission

1. Convert Markdown tables and equations to the current Electronics template.
2. Prepare editable source files and vector figures at journal-required sizes.
3. Generate a final PDF from the official template and inspect every page.
4. Confirm that the GitHub repository and archive are public.
5. Recheck the repository tag referenced by the Data Availability Statement.

## Claims deliberately excluded

- State-of-the-art accuracy.
- Universal superiority of GASF or multi-channel fusion.
- Generalization to arbitrary aggregate mains or unseen households.
- Real-time or edge-ready operation without measured latency, memory, and
  energy.
- Perfect Coffee maker/Water kettle recognition on data outside the fixed test
  split.
- Results from the exploratory `09_priority_baselines` directory before that
  code and output set is reviewed and integrated into the unified repository.
