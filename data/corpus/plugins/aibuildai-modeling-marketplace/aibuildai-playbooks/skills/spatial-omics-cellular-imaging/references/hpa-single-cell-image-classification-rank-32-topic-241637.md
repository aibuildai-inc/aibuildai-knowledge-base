# 32nd Place Solution

Competition: hpa-single-cell-image-classification
Rank: #32
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/241637

I had a great experience in the last months attempting to solve an interesting problem. Thank you all the participants for your great ideas and notebooks that helped me achieve my final results. I have learnt a lot from you. Thanks to the host team for always being supportive during the competition.

Inference Notebook: https://www.kaggle.com/quochungto/final-submission-hpa

### **OVERVIEW**

**Main components of my solution:**
- 2 stages pipeline - Cell level & Image level predictions
- Eliminating duplicate images - more reliable validation
- Faster version of cell segmentations
- External dataset
- Increasing diversity of models for ensembling: models trained on Green/RGB, 8bit/16bit images

As the competition stated, we are dealing with a weakly supervised semantic segmentation problem and only have labels for image level, not cell level. The task is to predict the label of each cell for an image with multiple cells.

Training single stage models to predict each cell’s label is difficult because we do not have the true target and using image labels as target will make it noisy to train. The idea is that each cell’s label inherits part of the image label, so using image prediction as context to correct each cell prediction will be effective compared to just predicting each cell independently.

### **TRAINING DETAILS**

#### **Dataset**
Competition dataset + External dataset

#### **Preprocessing**
As pointed out from other participants, the training dataset contained some duplicated images just like the last competition. Using this dataset directly in training will make validation less reliable. So the first thing I do is to try to remove all the duplicate images in the full dataset.

#### **Cell Segmentation**
All cell masks are extracted using the host's HPA Cell Segmentator as it does quite well in segmenting cells. Then each R, G, B channel is combined  into a single RGB image and saved as a new dataset. This reduces the problem to just weakly supervised multi-label classification.

#### **Models**
In order to maximize the effectiveness of ensembles, different backbones and inputs are used. This increases the diversity of my models.

**Cell level models**
- Input: Green/RGB cell images extracted from HPA Cell Segmentator
- Data augmentation: Horizontal & vertical flip, random crop
- Dataset: Competition & external datasets
- Epochs: 2 to 4
- Backbones: Resnet50, Efficientnet-b5
- Loss: BCEWithLogitLoss
- Optimizer: Adam
- TTA: 1 on original data + 4x

**Image level models**
- Input: Green/RGB images, 8-bit/16-bit images
- Data augmentation: Horizontal & vertical flip, rotate, shear, shift, zoom, random brightness
- Dataset: Competition & external datasets
- Epochs: 40 to 50
- Backbones: Resnet50, Densenet121, Efficientnet-b0, b1, b2, b3, b5, b7
- Loss: SigmoidFocalCrossEntropy
- Optimizer: Adam
- Learning rate scheduler: One-cycle
- TTA: 1 on original data + 4x

#### **Ensemble**
- Weighted average ensemble for each of cell level and image level prediction
- Final prediction as: `0.25 * cell_prediction + 0.25 * image_prediction + 0.5 * sqrt(cell_prediction * image_prediction)`

#### **Others**
Negative class:
- By replacing prediction of negative class with `P(neg) = prod(1 - P(cls_i))`, my public LB increased ~0.003

#### **Things that did not work**
- Hand labeling negative cell by calculating mean pixel intensity (public LB dropped)
- Eliminating border cells (no significant effect on public LB)
