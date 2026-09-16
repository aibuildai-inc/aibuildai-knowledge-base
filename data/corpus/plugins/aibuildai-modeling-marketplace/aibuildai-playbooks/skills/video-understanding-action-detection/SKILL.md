---
description: >-
  ML playbook for video understanding action detection competitions. Use when tackling a Kaggle-style competition involving video understanding action detection. Teaches how to reason about choose architecture family based on input modality and task structure, engineer invariant features for structured data before model input, design cross-validation to match test-set domain shift, apply modality-specific augmentations aggressively for regularization. 74 top-solution writeups across 7 competitions: ASL fingerspelling/signs (landmark-based sequence-to-text and classification), YouTube8M (temporal localization from features), MABe mouse behavior (pose-based detection), DFL, and NFL impact detection.
---

# Video Understanding Action Detection Playbook

Video understanding action detection tasks require recognizing and localizing actions in video data, which may be represented as raw frames, pre-extracted features, or pose/landmark sequences. The core challenge is modeling temporal dynamics while handling domain shift (different recording setups, signers, or environments) and often working under strict computational constraints (model size, inference time). Tasks range from isolated action classification to temporal localization to contact/interaction detection.

**Source material:** 74 top-solution writeups across 7 competitions: ASL fingerspelling/signs (landmark-based sequence-to-text and classification), YouTube8M (temporal localization from features), MABe mouse behavior (pose-based detection), DFL, and NFL impact detection.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Architecture Family Based on Input Modality and Task Structure | All top solutions matched architecture to data type. 7/7 competitions showed clear modality-architecture patterns. | principles/01.md |
| 2 | Engineer Invariant Features for Structured Data Before Model Input | 3/3 pose/landmark competitions (ASL fingerspelling, ASL signs, MABe) used hand-crafted invariant features. 1st place ASL fingerspelling: motion lag1/lag2. MABe 5th place: 58 engineered features including velocities, distances, angles. | principles/02.md |
| 3 | Design Cross-Validation to Match Test-Set Domain Shift | 5/7 competitions used stratified CV by a grouping variable. ASL: by participant_id (different signers). MABe: by lab_id. YouTube8M: video-level pre-train, segment-level fine-tune. | principles/03.md |
| 4 | Apply Modality-Specific Augmentations Aggressively for Regularization | All top solutions used heavy augmentation. ASL fingerspelling 1st: CutMix +0.005, FingerDropout +0.005, temporal stretch. MABe 2nd: Gaussian noise, time stretching, Mixup. | principles/04.md |
| 5 | Handle Variable-Length Sequences with Masking, Not Padding Alone | 5/5 sequence-based competitions required proper masking. ASL signs 1st place: 'Handling variable-length input correctly was very crucial for train-test consistency.' ASL fingerspelling used variable-length TFLite inference. | principles/05.md |
| 6 | Use Encoder-Decoder for Sequence Tasks, Direct Classification for Frame/Clip Tasks | 4/4 sequence-to-text or temporal localization competitions used encoder-decoder or CTC. ASL fingerspelling top 3: all encoder-decoder with transformer decoder. YouTube8M: video encoder → segment re-ranking. | principles/06.md |
| 7 | For Video Tasks, Encode Tracking or Metadata Directly into the CNN via Feature Channels | NFL impact detection solutions fed helmet positions to the CNN as extra input channels: 4th place added a heatmap of the helmet of interest and a heatmap of all helmet centers; 2nd place added all helmet boxes as channels for its second-stage model. | principles/07.md |
| 8 | Regularize Deeply with High Dropout, AWP, and Focal/Weighted Loss to Enable Long Training | 6/7 competitions used aggressive regularization. ASL signs 1st: dropout 0.8, AWP, drop_path 0.2. ASL fingerspelling 1st: 400 epochs. MABe 5th: dropout up to 0.35. | principles/08.md |
| 9 | For Temporal Localization, Use a Two-Stage Pipeline: Coarse Proposal → Fine Re-Ranking | 2/2 temporal localization competitions (YouTube8M) used two stages. 1st place: video model proposes top 100k videos → segment model re-ranks. 2nd place: video-level pre-train → segment-level fine-tune. | principles/09.md |
| 10 | Smooth Predictions Temporally via Neighboring Context or Post-Processing | 4/7 competitions used temporal smoothing. YouTube8M 1st: size-3 filter over time +0.01. MABe: sliding window with stride=half, average overlaps. | principles/10.md |
| 11 | Decode Sequence Outputs with Beam Search and Length Penalties, Not Just Greedy | 3/4 sequence generation competitions used beam search. ASL fingerspelling 4th: beam size 5-6 gave +0.005 over greedy. 2nd place: beam search for attention, greedy for CTC. | principles/11.md |
| 12 | Build Ensemble Diversity Through Architecture, Augmentation, and Fold Variation, Not Just Seeds | All top solutions ensembled. ASL fingerspelling 1st: 2 seeds with different training configs. YouTube8M 1st: 13+ diverse models. | principles/12.md |
| 13 | Post-Process to Align Predictions with Metric Quirks or Handle Corrupted Data | Several competitions used metric-aware post-processing. ASL fingerspelling 1st: replace low-confidence predictions with dummy phrase +0.006. 4th: corrupted data (< 50 frames) → constant prediction +0.005. YouTube8M 1st: filter predictions with non-decreasing function. | principles/13.md |
| 14 | Train with FP16 for Efficiency, But Verify Train-Inference Consistency | Several competitions used FP16. ASL fingerspelling 1st: 'Important to train with mixed precision to leverage fp16 inference without performance drop.' MABe 5th: FP16 training + inference. | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

Apply these principles in order: choose architecture based on modality, engineer features if data is structured, set up proper CV, then iterate on augmentation and regularization to enable deep models and long training. Ensemble diverse models and add metric-aware post-processing last.
