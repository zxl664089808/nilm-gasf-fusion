# Cover Letter

Dear Editor,

We submit our manuscript, **“Leakage-Aware Evaluation of Gramian Imaging, Raw Waveforms, and Physical Features for Plug-Load Appliance and State Recognition,”** for consideration as a research article in *Electronics*.

The manuscript provides a leakage-aware comparison of GASF images, V-I trajectories, raw voltage-current-power waveforms, and physical descriptors for high-frequency plug-load recognition. All 26-state comparisons use the same recording-grouped split, three model seeds, and classifiers near the 0.1-million-parameter scale. The study also removes rendered labels from the model input and separates source recordings and physical devices between data splits.

The main results are:

- The 11-class current-only GASF baseline reaches 94.23 ± 0.66% accuracy and 94.20 ± 0.67% macro-F1 under the unseen-recording protocol.
- Three-channel GASF imaging reaches 80.90 ± 6.61% image accuracy over three unseen-device splits.
- On the 26-state task, V-I trajectories reach 89.50 ± 0.68%, physical descriptors alone reach 94.99 ± 0.15%, and raw 3 x 500 V/I/P waveforms reach 96.95 ± 0.23%.
- Image-plus-physical fusion reaches 96.23 ± 0.23%, showing that most of its gain is attributable to the physical descriptors rather than the image conversion.
- A controlled image audit shows that class-dependent information remains accessible in the rendered title band after resizing.

The negative result for GASF accuracy is relevant to *Electronics* readers because it challenges a common time-series-to-image assumption and provides matched electronic-signal baselines for future compact recognition systems. The manuscript is original and is not under consideration elsewhere. The authors declare no conflicts of interest.

Sincerely,

Xianli Zeng  
Changsha University of Science and Technology  
Changsha, China  
zengxianli0626@163.com
