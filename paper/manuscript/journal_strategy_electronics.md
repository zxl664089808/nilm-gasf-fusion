# Journal strategy

## Position of the v0.3 manuscript

The manuscript is positioned as a leakage-aware electronic-signal recognition
study with three components:

- correction and audit of image-space label leakage;
- source-grouped and device-disjoint evaluation;
- compact GASF CNN and image-plus-physical-feature models for 11-class and
  26-label tasks.

The paper does not claim state-of-the-art performance or universal superiority
of image representations. Its contributions are reproducibility, complementary
representation analysis, and stable joint appliance/state recognition.

## First choice: Electronics (MDPI)

Fit:

- high-frequency voltage/current measurement;
- compact convolutional processing of electronic signals;
- practical ON/OFF and appliance recognition;
- interpretable electrical descriptors combined with learned features.

Required editorial controls:

- distinguish recording-level and device-level generalization;
- report class-specific failures rather than aggregate accuracy alone;
- state that latency, memory, and energy consumption were not measured;
- state that the task is event-aligned plug-load recognition, not arbitrary
  aggregate-mains disaggregation.

## Alternative: Pattern Analysis and Applications

This venue is appropriate if the paper emphasizes grouped pattern
classification, representation comparison, seed stability, and class-level
error analysis. The leakage audit and device-disjoint protocol would become
more central than the electronics application.

## Alternative: Signal, Image and Video Processing

This venue is appropriate if the paper emphasizes image encoding of
one-dimensional electrical signals, normalization effects, and compact CNN
classification. The manuscript would need a stronger signal-processing and
representation-focused introduction.

## Scope of the current submission

The v0.3 manuscript uses the committed and audited GASF experiments. Additional
raw-waveform, V-I trajectory, and physical-only exploratory runs that remain
outside the unified repository are not part of the submission claims. They can
be added in a later revision only after code review, result verification, and
integration into the same provenance-controlled repository.

## Remaining submission work

1. Confirm ORCID records and CRediT allocation.
2. Create a DOI-bearing repository archive.
3. Convert the manuscript and vector figures to the current Electronics
   template.
4. Recheck every number against the final tagged repository revision.
5. Upload the cover letter, manuscript, figures, and supplementary files.
