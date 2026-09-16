# 11th Place Solution

Competition: happy-whale-and-dolphin
Rank: #11
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/319916

Thank you for hosts, also team members(  @aerdem4   @ren4yu)
Competition is very tough, so we try to work harder every day, but it's very fun for us.
I write a summary of our team solution

# Summary
- Detection: YOLOV5 + WBF
- EfficientNet B5/B6/B7/V2S/V2M/V2L/V2XL + Pseudo Labeling
- Rescore Siamese Network

# Detection Part
## Dataset
Firstly, We found small whale in images
so we need to focus on whale for accurate identification.

I use labelimg for annotating whales. this annotation tool can export yolo format.
https://github.com/tzutalin/labelImg

Finally, we annotated 5800 images.

## Training
- img size: 1280
- YOLOV5x6
- BS8
- SyncBN
- Epoch 20
- 6 Fold

## Inference
6fold models + WBF -> Filter top 1box

# Whale Identity Part
## Dataset
- Detection result

## Training
- EfficientNet B5/B6/B7/V2S/V2M/V2L/V2XL
- ArcFace
- Adam
- Pseudo labeling(Multi-step, Threshold)
- CosineAnnealingWarmRestarts
- GPU/TPU
 
ensemble using concat(32000dim). a single model is about 0.805.
I split 100folds for training.

## Prediction
We use cosine similarity for the identification of whales with some post-process.

## Rescore
Before prediction, We use Siamese Network for top20
We combine the similarity matrix and Siamese Network Score. it's a huge improvement in our score.
It's achieved Public 0.881/Private 0.853

# we didn't work
- DoLG
- twice identity using Flip
- Swin/ConvNeXt
