# 8th place solution

Competition: landmark-retrieval-2020
Rank: #8
Source: https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176405

First, thanks to Google and Kaggle for hosting this competition and congratulations to the winners :)

Since our solution is quite similar to the ones already posted and to DELG itself, I'll keep it simple and brief

### Dataset
- GLDv2 clean
- 80% training, 20% val split by image

### Loss
- ArcFace Layer
  - Margin: 0.3
  - Scale: 46

### Models
- ResNet101
- EfficientNetB5
- GeM pooling
  - p=3 frozen for R101
  - p trained for B5
- 2048d descriptors by applying FC + BN after pool

### Training
- Trained until convergence at 512x512
- Fine-tuned for a few epochs at 640x640
- Around 35 epochs for R101 and 20 for B5

### Inference
- Multi-scale TTA
  - R101: Resize to (640, 768) squared images 
  - B5: Resize to 640 squared images + resize to 1024 preserving AR
  - 2048d descriptors per model by averaging multi-scale predictions followed by l2-normalization
- Ensembling by concatenating model's predictions into a 4096d descriptor followed by l2-normalization
