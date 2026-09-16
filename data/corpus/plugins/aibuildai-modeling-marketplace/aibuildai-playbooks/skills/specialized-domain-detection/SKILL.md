---
description: >-
  ML playbook for specialized domain detection competitions. Use when tackling a Kaggle-style competition involving specialized domain detection. Teaches how to reason about model both raw signal and domain-transformed representations, preserve high resolution in early layers when signal is spatial-frequency, use larger crops to capture contextual information, design depth-invariant architectures for volumetric data. Analysis of 37 top-solution writeups across 10 competitions: deepfake detection, camera model identification, power line fault detection, face recognition, audio tagging, scientific image forgery, and malware classification.
---

# Specialized Domain Detection Playbook

Specialized domain detection encompasses tasks where subtle, domain-specific signals must be distinguished from natural variation—identifying camera models from sensor noise, finding duplicated regions in scientific figures, detecting partial discharge in power-line signals, or spotting deepfakes. The core challenge is that the signal of interest is often imperceptible to humans and easily destroyed by standard preprocessing, requiring domain knowledge to preserve the evidence while building models robust enough to generalize.

**Source material:** Analysis of 37 top-solution writeups across 10 competitions: deepfake detection, camera model identification, power line fault detection, face recognition, audio tagging, scientific image forgery, and malware classification.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Model Both Raw Signal and Domain-Transformed Representations | Camera model ID 10th (image CNNs stacked with sensor-noise feature models), audio tagging 4th/8th (raw waveform, log-Mel and MFCC inputs; spectrogram CNN joined with spectral statistics), scientific image forgery 29th (image plus noise-residual input) | principles/01.md |
| 2 | Preserve High Resolution in Early Layers When Signal is Spatial-Frequency | Camera model ID solutions train on native-resolution crops instead of resized images (1st, 3rd, 4th, 5th); scientific image forgery 29th adds a stride-1 noise residual before the first pooling layer | principles/02.md |
| 3 | Use Larger Crops to Capture Contextual Information | Deepfake 1st/2nd (face crops with a large margin around the face), camera model ID 3rd (train on small patches, fine-tune on larger ones); camera model ID 1st/5th found small crops nearly as accurate and much faster | principles/03.md |
| 4 | Design Depth-Invariant Architectures for Volumetric Data | Passenger screening 8th (one 2D encoder shared by all 16 views, features averaged across views); passenger screening 1st concatenated the views instead because threat location mattered, and 4th found a 3D model worse than its 2D model | principles/04.md |
| 5 | Apply Domain-Specific Augmentations Beyond Standard Transforms | Deepfake 1st/3rd (video compression, face blending, mixup on aligned pairs), camera model ID 4th (the competition's own image manipulations), scientific image forgery 1st (simulated figure artifacts), power line 26th (cyclic shift of a periodic signal) | principles/05.md |
| 6 | Trust Local Cross-Validation Over Public Leaderboard | Camera model ID 1st/3rd (tuning to the public LB overfit; trust your validation), power line 9th/38th (picked the stable or conservative model over the best public score), scientific image forgery 1st (built a clean validation set) | principles/06.md |
| 7 | Train Multi-Head Outputs (Binary + Multiclass) Simultaneously | Deepfake 2nd (classification branch plus a segmentation head trained on real-fake difference masks, about 0.01 LB gain); passenger screening 4th (one detector label encodes both the threat and its body zone) | principles/07.md |
| 8 | Stack Diverse Models with a Second-Level Meta-Learner | Camera ID 1st/10th (52-model blend, XGBoost stacker), audio tagging 4th (meta-learning ensemble), passenger screening 1st/4th (gradient-boosted models trained on out-of-fold or holdout scores) | principles/08.md |
| 9 | Ensemble Models by Averaging Predictions Before Sigmoid/Softmax | Deepfake 41st (average the scores of all networks and frames, then apply the sigmoid); camera ID 2nd and audio tagging 8th (geometric mean of predictions, an average in log space) | principles/09.md |
| 10 | Apply Aggressive Test-Time Augmentation with Spatial and Domain-Specific Transforms | Camera ID 1st/2nd/5th (TTA8; five crops with flips or the D4 group; manipulations and rotations), passenger screening 1st (about 100 augmented predictions per scan), audio tagging 8th (predictions over several clip lengths and random spans) | principles/10.md |
| 11 | Clean Up Predictions with Connected-Component Post-Processing | Deepfake 5th (kept only faces that stay connected across space and time for at least 30 frames) and 47th (tracked faces across frames to remove spurious detections) | principles/11.md |
| 12 | Collect and Curate External Data When Allowed | Camera ID 1st place (300GB Flickr/Yandex photos), 2nd place (large external dataset), deepfake solutions (careful avoidance due to rules) | principles/12.md |
| 13 | Engineer Domain-Specific Features for Signal/Time-Series Data | VSB power line 2nd place (peak features from scipy.signal), freesound audio 4th (MFCC, delta-delta) | principles/13.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together as a system: multi-domain modeling provides complementary views, high-resolution preservation and large crops retain the signal, and domain augmentations and diverse ensembles prevent overfitting. Always validate each technique on a rigorous local CV setup and trust it over a small public leaderboard.
