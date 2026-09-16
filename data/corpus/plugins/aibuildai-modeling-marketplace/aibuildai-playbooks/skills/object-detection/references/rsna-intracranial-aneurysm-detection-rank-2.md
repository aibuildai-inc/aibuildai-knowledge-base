# 2nd Place Solution

Competition: rsna-intracranial-aneurysm-detection
Rank: #2
Source: https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution

Thanks to @evancalabrese, RSNA, and Kaggle for organizing this Intracranial Aneurysm Detection competition! A special shoutout to the insightful discussions (especially @honglihang's detailed post on multi-frame DICOM analysis) that helped us gain a deeper understanding of the competition data.

## Overview

Our approach focused on simplicity and generality to handle the diverse data in this classification-focused task. Key elements:

- **Stage 1**: Fast 2D tri-axial ROI extraction using an nnU-Net 2D segmentation model to crop binary vascular regions efficiently, validated on all training cases locally.

- **Stage 2**: 3D multi-task learning (segmentation of vessels and aneurysms + classification) based on nnU-Net, with enhancements like cross-attention pooling, modality 4-class heads, and targeted oversampling for rare classes. All data resized to uniform 224x224x224; heavy TTA (8x) including left-right flips with label swapping.

Built entirely on the open-source nnU-Net framework—big thanks to @fabianisensee and team for its robust 3D segmentation baseline.

Inference optimized for speed (encoder + classification head only), running on 2x T4 GPUs in ~9 hours for conservative ensembles.

## Who We Are

We're a mix of algorithm engineers and PhD researchers: Pengcheng Shi, Yan Lu, and Jiawei Chen from Medical Image Insights in Shanghai. Kaiyuan Yang and Houjing Huang from UZH in Zurich. Kaiyuan Yang, Houjing Huang, and Pengcheng Shi are also organizers of the MICCAI TopCoW (https://topcow24.grand-challenge.org/) / TopBrain (https://topbrain2025.grand-challenge.org/) challenges that benchmarked the segmentation of Circle of Willis (CoW) and whole brain vessel anatomy. Most of us are new to Kaggle (our background is in MICCAI events), and we noticed differences: Kaggle emphasizes data wrangling, efficiency under resource limits, and flat metrics (pure classification here, no localization/segmentation scores). In the TopCoW summary pre-print, we have previously explored the potential of TopCoW segmentation model at locating aneurysm with the CoW anatomy, which inspired many ideas used in our current solution.

## Data Handling Challenges

Kaggle's data diversity (spacing, modalities, multi-frame DICOMs) required heavy preprocessing. Multi-frame issues were tricky—we had limited DICOM experience, and test sets had deleted fields, breaking dicom2nifti. We switched to pydicom, mapping spacing by slice shapes (e.g., <45 slices → 5mm), and trained a T2-specific orientation classifier for corrections. This ate up time but ensured full test coverage. No try-except fallbacks to 0.5 predictions in final inference—maximized robustness.

We used flipping TTA and multi-fold model ensembling to increase robustness of the model. Promising results showed that increasing in the public leaderboard led to consistent improvement in the private leaderboard.

## Stage 1: 2D Tri-Axial ROI Extraction

To crop binary vascular regions efficiently:

- Sample 3 slices per axis (1/4, 1/2, 3/4 positions) from iterative vascular ROI data → 9 slices total.
- Train nnU-Net 2D config on sliding windows for cropped segmentation.
- Inference: Merging sliding window patches into the batch dimension enhances both speed and robustness.
- Local tests: Handled 100% of training cases correctly.
[stage1]
This stage was key for quick, universal ROI on varied data.

## Stage 2: 3D Multi-Task Learning

Unified all spacing/modalities by resizing to 224x224x224.

**Architecture**: nnU-Net 3D with multi-task heads (vessel/aneurysm seg + classification).

**Enhancements**:

- Cross-attention pooling surpassed average pooling in convergence speed.
- Modality classification head enhanced training stability.
- Inference: Only encoder and class head forwarded to reduce latency.
- Data aug: Left-right flips with swapped labels and masks for classification and segmentation, effectively doubling dataset size.
[vessel_flip]
*Left: Original vascular anatomy segmentation mask | Right: Augmented mask after horizontal flipping with swapped labels*
- Aneurysm masks were iteratively generated via model inference and manual refinement based on aneurysm center points.
- 13-class vascular segmentation refined aneurysm masks using center-point distance heatmaps.
- Rare classes were oversampled and assigned higher cross-entropy weights for both classification and segmentation.
- Vessel and aneurysm segmentation stabilized classification convergence.
- TTA: 8x (flips; swap left/right labels on outputs).
- Final submissions: Due to platform constraints, we cut model count and ran a conservative two-fold ensemble on 2×T4 GPUs to prevent timeouts (runs were ~9 hours).
[stage2]
## Training Details

Based on nnU-Net defaults, trained from scratch.

**Loss**: CE with heatmap weighting for aneurysm centers.

**Augmentations**: Standard nnU-Net (rotations, scaling, noise) + custom left-right flips and label swaps.

**External data**: 
- TopCoW Training Data and its External Testsets: https://zenodo.org/records/15692630 (Note: The LargeIA dataset was excluded from training.)
- TopBrain annotations: https://zenodo.org/records/16878417

**Training data annotation and correction**: 
- Aneurysm seg mask: Center position provided in the challenge data csv combined with light manual annotation to iteratively annotate and train our own aneurysm segmentation model.
- The provided rsna aneurysm annotations contain a few cases that mixed up left vs right side aneurysms, supra- vs infra-clinoid ICA aneurysms,and other position label mix-ups, which were manually corrected if identified.
- Vessel seg mask customized by merging provided cow-seg masks with TopCoW and TopBrain annotations; a few hard cases, especially of T2 and T1-post (some T1-post even have artery flow void), were manually corrected.

## Results and Insights
### Ablation Study
| Experiment Configuration | Public Score | Private Score | Notes |
| :----------------------- | :----------- | :------------ | :---- |
| Baseline | 0.84407 | 0.81268 | • First version segmentation annotation data<br>• Epoch 100, Fold 0<br>• resize 224×224×224<br>• cls_modality |
| Baseline + TTA 4x | 0.87056 | 0.82508 | • First version segmentation annotation data<br>• Epoch 100, Fold 0<br>• resize 224×224×224<br>• cls_modality<br>• Test-Time Augmentation 4x |
| Improved Data + TTA 4x | 0.88832 | 0.85718 | • Improved segmentation annotation data<br>• Epoch 100, Fold 0<br>• resize 224×224×224<br>• cls_modality<br>• Test-Time Augmentation 4x |
| Improved Data + TTA 8x | 0.89805 | 0.86228 | • Improved segmentation annotation data<br>• Left-right data augmentation with label swap<br>• Epoch 100, Fold 0<br>• resize 224×224×224<br>• cls_modality<br>• Test-Time Augmentation 8x |
| Improved Data + TTA 8x + Ensemble | 0.90035 | 0.86727 | • Improved segmentation annotation data<br>• Left-right data augmentation with label swap<br>• Epoch 100, Fold 0 + Epoch 250, Fold 1 (Ensemble)<br>• resize 224×224×224<br>• cls_modality<br>• Test-Time Augmentation 8x |

### Key Technical Insights

- ROI Extraction: Our Stage 1, 2D tri-axial ROI extraction significantly improved inference speed.
- Vessel Segmentation: Incorporating vessel segmentation clearly enhanced aneurysm segmentation and classification performance.
- Noise Handling: Our noise-handling techniques performed more effectively on the public dataset.
- Dataset Shift: Removing the 0.5 prediction fallback revealed a consistent 3–4% performance gap between the public and private leaderboards. This suggests a differing distribution of abnormal cases between the two test sets.

### Performance Evolution

- A consistent performance improvement was observed across all experiment iterations.
- Top Result: "Improved Data + TTA 8x + Ensemble" scored **0.90035** (Public LB, 1st) and **0.86732** (Private LB, 2nd).

## Acknowledgements

Thanks to Medical Image Insights and UZH for compute support, Bjoern Menze and the Helmut Horten Foundation for funding support. We are grateful to RSNA/Kaggle hosts, nnU-Net devs, and forum contributors.

## Code Availability

- Training and inference code: https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection
- Inference demo: https://www.kaggle.com/code/pengchengshi/bravecowcow-2nd-place-inference-demo
- Final submission inference: https://www.kaggle.com/code/pengchengshi/bravecowcow-2nd-place-inference-final-submission
- Stage 1 model checkpoint: https://www.kaggle.com/models/pengchengshi/dataset180_2d_vessel_box_seg_stable
- Stage 2 model checkpoint: https://www.kaggle.com/models/pengchengshi/rsna2025-stage2-models

## Citation
Please cite our work if it is helpful for your research:

```
@misc{shi2026intracranialaneurysmclassificationsegmentation,
      title={Intracranial Aneurysm Classification and Segmentation via Tri-Axial ROI and Multi-Task Learning}, 
      author={Pengcheng Shi and Kaiyuan Yang and Houjing Huang and Jiawei Chen and Yan Lu and Jiaqi Liu and Murong Xu and Bjoern Menze and Xinglin Zhang},
      year={2026},
      eprint={2606.26706},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.26706}, 
}
```

## References

- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203-211.
- Yang, Musio, Ma, et al. "Benchmarking the CoW with the TopCoW Challenge: Topology-Aware Anatomical Segmentation of the Circle of Willis for CTA and MRA." arXiv (2025): arXiv-2312
