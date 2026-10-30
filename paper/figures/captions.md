# Figure captions

**Figure 1.** Event-aligned multi-physical GASF recognition pipeline. Voltage
and current cycles are segmented at upward voltage zero crossings. Voltage,
current or differential current, and power are encoded as GASF channels. The
image-only model uses the three-channel tensor directly; the fusion model
additionally uses 14 physical descriptors.

**Figure 2.** Image-space leakage audit. (a) Historical image with the class and
state rendered in the title band. (b) Corrected image containing only numerical
RGB pixels. (c) Diagnostic classification using only the top 12 rows of resized
title-bearing and title-free images under an image-level split.

**Figure 3.** Unseen-recording channel comparison for 11-class recognition.
Bars show mean test accuracy and macro-F1 over seeds 42-44; error bars show one
standard deviation.

**Figure 4.** Device-level evaluation of the 11-class models. Panel (a) shows
image-weighted accuracy and panel (b) shows device-balanced accuracy. Image
V+I+P and Hybrid V+I+P use three device splits; the I configurations use the
first device split.

**Figure 5.** Model comparison for 26-label recognition. Bars show mean test
accuracy and macro-F1; error bars show one standard deviation over seeds 42-44
where applicable. Multitask is a single seed-42 run.

**Figure 6.** Recall for difficult classes in the 26-label task. Bars compare
the image-only V+I+P model with physical-feature fusion. Values are means over
seeds 42-44, and error bars show one standard deviation.
