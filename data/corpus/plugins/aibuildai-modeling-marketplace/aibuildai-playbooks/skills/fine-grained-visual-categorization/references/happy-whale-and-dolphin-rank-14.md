# 14th place solution

Competition: happy-whale-and-dolphin
Rank: #14
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/320407

Thanks to the organizers for hosting the interesting competition and congratulations to all the winners. It was tough, but was a great experience for me.

## Datasets

Three Yolov5 models are trained to detect fullbody and backfin, respectively.
Scores improved greatly with changing datasets as: original image -> detic crop -> fullbody crop -> fullbody/backfin crop.

## Models

- ArcFace models are trained using fullbody crop (640×640) and backfin crop (448×448), respectively. The parameters of ArcFace are set as scale = 25 and margin = 0.5 (The large margin worked with improved datasets in my case). Embedding size is 2048.
- Focal loss for species classification (loss weight = 0.1) is also used.
- Backbone: Efficientnet-b7 and ConvNeXt-L
- In order to make embeddings more discriminative, multiply the feature maps by the attention weights computed with GAP features as a query. This worked only with Efficientnet.
- Pseudo labeling: Take about 30% top predictions (would be better to use more data and multiple iterations).
- Distillation: Use the features of the teacher model as soft target and compute MSE. This greatly improves the performance in the early stages of training, but the contribution to the final score does not seem to be that large, so adopt cosine schedule to set the final loss weight to 0.

### Augmentation

Average blur, motion blur, gaussian noise, saturation, brightness, contrast, grayscale and affine transform (flip, rotation, shear, scaling and translation) are used.

### Training details

- Hyper parameters: AdamW with weight decay = 0.05, base lr = 2e-4, cosine scheduling with linear warmup, batch size = 24 along with gradient accumulation
- PyTorch is used and training 30 epochs with a single A100 GPU take about 30 hours for a fullbody model and 15 hours for a backfin model.
- Private LB of the best single fullbody model is 0.781.

## Post-process

1. Cosine similarity matrix for the individuals is computed using the features of the test images and ArcFace weights using each model.
1. Then average the matrices to ensemble eight fullbody/backfiin models.
1. Insert new_individual with a fixed threshold (should have changed threshold for each species).
1. Finally, switch adjacent top predictions based on the difference of cosine similarity and the degree of assignment to the top for class balancing.

## What did not work

The following did not work in my case.

- Elasticface
- Magface
- Setting sample-wise margin based on the magnitude of losses
- Triplet loss
- Semi-supervised learning with self-distillation loss of DINO (somewhat worked, but employed pseudo-labeling)
- Adding a SOD mask channel to input
- Adding an attention weight channel to input

Thank you for reading.
