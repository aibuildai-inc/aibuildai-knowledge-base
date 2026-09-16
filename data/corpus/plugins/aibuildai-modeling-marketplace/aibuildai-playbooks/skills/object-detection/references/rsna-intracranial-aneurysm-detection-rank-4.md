# 4th Place Solution

Competition: rsna-intracranial-aneurysm-detection
Rank: #4
Source: https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution

Sincere Thank You to RSNA and Kaggle for hosting this competition 

I tackled the challenge as a speedrun as I had only 14 days left on the competition when I started, which is nowhere near enough to cover everything one can do with this dataset, the dataset is super cool and I have barely scratched the surface. I managed to get in the gold medal line in 8 days.

Finally a GM, yay!

### Segmentation Model Failure

I started with training a segmentation model… which was not really happy to learn beyond 0.6 Dice in the short time I tried which was nowhere near enough for my confidence in the approach.

### ROI Crop coordinates Model

I shifted to a ViT (dinov3) model that takes in input of equidistant 48 slices for each patient, input shape: 48x1x128x128 volume and predicts x1, x2, y1, y2 values of where the crop boundaries of segmetation mask should be at patient-level, this model was far more reliable and efficient for ROI cropping

Original:



ROI Crop:



I kept the ROI relatively large as I did not want to miss out on aneurisms, I tested with the coordinate data and doing it in this configuration let me keep 95%+ of aneurism locations inside the cropped region

### Classification Model

I took ~2.2k samples in the train_localizer csv file and all samples from negative patients which gives me a dataset of:
- 545k samples with ~2.2K positive samples and rest negative samples (1 in 250 positive), 
- There are 14 labels in the dataset, which keeps it that model needs to predict 1/1700 values to be 1, rest 0s

I managed to find a combination of pipeline where model starts to learn even in this highly imbalance without weighted sampling (weighted sampling did not help).

My entire solution is this classification model and here’s how I improve it’s score (very simple once baseline starts learning)

My baseline hyperparameters:
- Model: CoaT-Lite-Medium
- Optimizer: AdamW | Learning-Rate: 1e-4 cosine annealing down to ~3e-5
- Augmentations: HFlip, VFlip, no TTA
- Image size: 384x384
- Global Batch-Size: 192

| Approach | CV | Public/Private Leaderboard |
|---|---|---|
| CoaT-Lite-Medium on ROI Crops | 0.805 | 0.78/0.78 |
| + Rotate (-25, 25) | 0.83 | 0.8/0.81 |
| + 2.5D (-2, 0, +2) | 0.86 | 0.86/0.83 |
| + Soft Pseudo-Distill on remaining 500k potentially positive samples | 0.89 | 0.86/0.84 |
| Ensembling with MaxViT + CoaT | 0.896 | 0.87/0.84 |

<br>

On inference, the model predicts on every image for each patient, and final predictions are simply max aggregated

**Thank you for reading!**


Inference Code [here](https://www.kaggle.com/code/harshitsheoran/rsna-aneurysm-detection-demo-submission)
Training Code [here](https://www.kaggle.com/datasets/harshitsheoran/rsna2025-training-code)
Model Weights [here](https://www.kaggle.com/datasets/harshitsheoran/rsna2025-raw-models)
