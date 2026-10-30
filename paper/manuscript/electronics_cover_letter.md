# Cover Letter

Dear Editor,

We submit our manuscript, **"Leakage-Aware Multi-Physical GASF Imaging for
Plug-Load Appliance and State Recognition,"** for consideration as a research
article in *Electronics*.

The manuscript investigates how high-frequency electrical signatures can be
represented and evaluated without the label and source leakage that can
inflate image-based NILM results. We remove all rendered text from the model
input, group correlated cycles by source recording, and supplement the 11-class
evaluation with device-disjoint validation. For the 26-label task, we compare a
three-channel GASF CNN with a compact image-plus-physics model using the same
recording-grouped split and three training seeds.

The principal results are:

- A current-only GASF image model reaches 94.23 +/- 0.66% accuracy and
  94.20 +/- 0.67% macro-F1 for 11-class unseen-recording recognition.
- The three-channel image model reaches 80.90 +/- 6.61% image-level accuracy
  and 90.62 +/- 2.76% mean device accuracy over three unseen-device splits.
- Adding 14 physical descriptors increases 26-label mean test accuracy from
  87.06 +/- 0.13% to 96.23 +/- 0.23% and macro-F1 from 85.72 +/- 0.17% to
  96.23 +/- 0.24%.
- The physical branch removes the severe Coffee maker/Water kettle confusion
  under the fixed held-out split while exposing persistent weaknesses in
  Fridge_OFF and selected lighting states.

The work is relevant to *Electronics* readers because it connects electronic
signal conditioning, compact CNN design, and leakage-aware validation for
appliance and state recognition. It provides reproducible preprocessing and
explicitly distinguishes unseen-recording from unseen-device performance. The
manuscript is original, is not under consideration elsewhere, and the authors
declare no conflicts of interest.

Sincerely,

Xianli Zeng

Qinghua Zhou (Corresponding author)

Changsha University of Science and Technology, Changsha, China
