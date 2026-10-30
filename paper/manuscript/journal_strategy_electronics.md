# Journal strategy after representation controls

## Evidence-driven positioning

The matched experiments show that raw 3 x 500 V/I/P waveforms outperform the
tested 64 x 64 GASF images on the 26-state task. The manuscript must therefore
be positioned as a leakage-aware representation evaluation rather than as a
claim that Gramian imaging improves accuracy.

## First choice: Electronics (MDPI)

Use this positioning only if the manuscript emphasizes:

- high-frequency electronic-signal measurement;
- matched raw-waveform, V-I, image, and physical-feature comparisons;
- compact classifiers with approximately 0.1 million parameters;
- leakage-aware validation and practical device-shift limitations.

The paper should state explicitly that GASF did not improve accuracy in the
completed experiment. That negative result is a contribution only if supported
by the matched protocol and full baselines.

## Alternative: Pattern Analysis and Applications

This venue may fit the paper better if the central contribution is:

- representation comparison;
- leakage-controlled pattern-classification methodology;
- seed and device robustness;
- class-specific error analysis.

The risk is that the method combines established components. The benchmark and
negative result must therefore be the main contribution.

## Alternative: Digital Signal Processing

This venue is suitable if the paper emphasizes:

- raw sequence versus image-domain representation;
- normalized V-I trajectory encoding;
- amplitude loss caused by Gramian normalization;
- comparison against physical descriptors and raw 1D CNNs.

## Before submission

1. Confirm ORCID records and the CRediT statement.
2. Decide whether one full ResNet-18 capacity control is necessary.
3. Add actual latency and memory measurements if an embedded or edge claim is
   desired.
4. Provide a code and data archive.
5. Reformat the manuscript and references to the chosen journal template.
