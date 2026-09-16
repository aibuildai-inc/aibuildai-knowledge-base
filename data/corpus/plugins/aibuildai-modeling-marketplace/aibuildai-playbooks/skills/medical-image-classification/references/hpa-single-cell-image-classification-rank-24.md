# 24th place (private) solution - Single model approach

Competition: hpa-single-cell-image-classification
Rank: #24
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238401

In this competition, multiple different approaches like cell level classification or image level heat-map prediction both seemed possible. During the early phase of the competition, I tried detection models (mmdetection and yolo) using the raw image level labels. But failed get a good score. This made me assume that image level labels are far to represent labels for each cell.

Moving forward, I wanted to try CAM based models. Puzzle-CAM paper and their implementation was easy to follow. My final model was based on Puzzle-CAM without much modification of training schema or model architecture.

#  Hyperparameters
**Augmentation:**
Random crop of ~80% of input image size
Random flip/transpose
Random brightness/contrast
Random cutout
**Optimizer:** AdamW
**Epoch:** 10
**Loss:** Multilabel soft-margin loss for classification + L1 loss for reconstruction
**Learning rate:** linearly decreasing from 1e-4 to 1e-6

# Inference
Probability for each cell was calculated by multiplying image level probability with normalized CAM logits. Sliding window was used to predict on input image.
TTA: 2 times flip and 3 scales (0.75, 1.00, 1.50)

# Ablation study
Model: Densenet121 
Input resolution: 800x800 -> 512x512 crop
Dataset: Only competition 
LB: 0.457 (public)
Dataset: Competition + External data 
LB: 0.481 (public)
Input resolution: 1280x1280 -> 1024x1024 crop
LB: 0.503(public)

Model: Xception
LB: 0.527(public)
LB: 0.495(private)


I have tried cell level classifier by cropping individual cells and training with the labels found from CAM model. But the labels were still noisy and didn't produce better results than CAM models. So cell level models were not used in the final solution.

I have also tried to ensemble CAM based models. Trained Densenet121, Resnest50 and Xception models (each single fold). Xception performed better than others and ensemble provided a little gain. So decided to go with single model (xception) and TTA.

I did experiment with a few loss functions. Focal Loss, Asymmetric Loss For Multi-Label Classification and Sharpness-Aware Minimization are some of them. They are all good fit for this problem, but probably needed more hyperparameter tuning.

Inference code: [Xception-0.495 private](https://www.kaggle.com/sgalib/0-495-private-single-model-puzzlecam)

Thanks for reading!
