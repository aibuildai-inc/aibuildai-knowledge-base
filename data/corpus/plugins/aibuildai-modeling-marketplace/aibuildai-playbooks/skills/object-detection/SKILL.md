---
description: >-
  ML playbook for object detection competitions. Use when tackling a Kaggle-style competition involving object detection. Teaches how to reason about choose detector architecture based on object size and aspect ratio distribution, design train/val split to match test distribution, especially for sequential data, apply heavy augmentation with mixing techniques for small datasets, train and infer at multiple resolutions to balance speed and small object recall. Analysis of 99 top-solution writeups (ranks 1-163) across 13 Kaggle object detection competitions including Global Wheat Detection, TensorFlow Great Barrier Reef, RSNA Intracranial Aneurysm Detection, and Severstal Steel Defect Detection.
---

# Object Detection Playbook

Object detection requires localizing and classifying objects within images, typically outputting bounding boxes with class labels and confidence scores. Unlike pure classification, success depends on precise spatial localization measured by Intersection over Union (IoU) thresholds. The core challenge is balancing recall (finding all objects) with precision (avoiding false positives) across varying object scales, occlusions, and domain-specific characteristics like annotation quality, temporal sequences, and distribution shifts between train and test sets.

**Source material:** Analysis of 99 top-solution writeups (ranks 1-163) across 13 Kaggle object detection competitions including Global Wheat Detection, TensorFlow Great Barrier Reef, RSNA Intracranial Aneurysm Detection, and Severstal Steel Defect Detection.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Detector Architecture Based on Object Size and Aspect Ratio Distribution | Top solutions consistently consider object characteristics when selecting models. YOLO variants dominate among top-3 solutions, but anchor-free methods (CenterNet, FCOS) excel for extreme aspect ratios. | principles/01.md |
| 2 | Design Train/Val Split to Match Test Distribution, Especially for Sequential Data | Most top solutions in video-based competitions used sequence-aware splits. In non-sequential tasks, several winning solutions stratified by metadata (source, image characteristics). | principles/02.md |
| 3 | Apply Heavy Augmentation with Mixing Techniques for Small Datasets | Nearly all top solutions used mosaic, mixup, or cutmix; mosaic was the most common, followed by mixup. The 1st place Global Wheat solution gained +0.03 mAP from custom mosaic alone. | principles/03.md |
| 4 | Train and Infer at Multiple Resolutions to Balance Speed and Small Object Recall | 12/13 top solutions in competitions with multi-scale objects used different train/inference resolutions. Great Barrier Reef 2nd place: training at 2400px, inference at 4800px improved LB by +0.13 F2 despite hurting CV. | principles/04.md |
| 5 | Leverage Pseudo-Labeling on Test Data When Competition Permits Code Submissions | Most code competitions' top solutions used pseudo-labeling. Global Wheat 1st place improved from 0.7629 to 0.7656 (+0.0027) with 2 rounds. Severstal 1st place: +0.002 from pseudo labels, noting timing matters—too early hurts. | principles/05.md |
| 6 | Ensemble Diverse Model Architectures with Weighted Boxes Fusion (WBF) | WBF was the standard ensembling method in the box-detection competitions: all 7 Global Wheat writeups and 13 of 17 Great Barrier Reef writeups use it. Hydrogen team (Great Barrier Reef 3rd): 5 different architectures ensembled with WBF. | principles/06.md |
| 7 | Investigate and Adapt to Annotation Quality Differences Between Train and Test | RSNA Aneurysm 2nd place manually corrected training labels that mixed up left and right sides and other positions; its improved annotation data raised the private score from 0.825 to 0.857 in its ablation. Great Barrier Reef 3rd place: test labels were 3px tighter on all sides. | principles/07.md |
| 8 | Apply Temporal Tracking for Video Sequences to Boost Confidence of Persistent Objects | Nearly all video-based competition top solutions used tracking. Great Barrier Reef 1st place: 'attention area' tracking boosted CV by +0.01. 3rd place: tracking low-confidence boxes via trajectory improved recall without adding false positives. | principles/08.md |
| 9 | Use Tiled Training and Inference for Small or Densely Packed Objects | 4/6 solutions in competitions with very small objects (<50px mean) used tiling. Great Barrier Reef 9th place: 320px tiles improved F2 by +0.014 vs full-image training. Global Wheat 30th place: quadrant detection with overlap boosted mAP +0.03. | principles/09.md |
| 10 | Tune NMS IoU Threshold Lower Than Default for Dense or Overlapping Objects | Several solutions in dense-object competitions lowered NMS IoU threshold. Great Barrier Reef solutions commonly used 0.3-0.4 for clustered starfish. | principles/10.md |
| 11 | Add a Classification Head to Re-Score Detections for Multi-Class Datasets with Class Imbalance | Several RSNA Aneurysm top solutions detected candidate regions first and then classified them (15th place: Faster R-CNN RoIs into a Transformer; 9th place: YOLO outputs into meta-classifiers). Great Barrier Reef 1st place: classification re-scoring boosted CV from 0.716 to 0.73+. | principles/11.md |
| 12 | Clean Training Data by Removing or Correcting Obvious Annotation Errors | Several top solutions mentioned data cleaning. Global Wheat 1st place removed boxes <10px and fixed oversized boxes. Great Barrier Reef 9th place relabeled inconsistent/missing labels. | principles/12.md |
| 13 | For 3D or Volumetric Detection, Use Coarse-to-Fine Pipelines to Manage Computational Cost | 3/3 top solutions in 3D medical imaging (RSNA Aneurysm, BYU Flagellar Motors) used multi-stage coarse-to-fine. RSNA 1st place: low-res localization → high-res ROI segmentation → classification saved 60% inference time. | principles/13.md |
| 14 | Use Test-Time Augmentation (TTA) with Geometric Transforms, Not Just Flips | Most top solutions used TTA beyond horizontal flip. Global Wheat 1st place: HFlip + VFlip + Rotate90 gave +0.01 mAP. Great Barrier Reef 2nd place: 11 TTA combinations (scales + rotations) improved ensemble diversity. | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

Apply these principles in sequence: start with architecture and split strategy (foundation), then layer in augmentation and multi-scale techniques (training robustness), and finally optimize inference with ensembling, TTA, and post-processing (performance ceiling). Always validate each technique on your specific competition metric and data characteristics—what works for wheat detection may not apply to medical imaging, and vice versa.
