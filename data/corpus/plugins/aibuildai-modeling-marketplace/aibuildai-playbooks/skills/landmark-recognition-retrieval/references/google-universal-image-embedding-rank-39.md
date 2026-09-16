# 39th place solution

Competition: google-universal-image-embedding
Rank: #39
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359798

Thank you for competiton hosts and competitors!

This is 39th place solution.

# Model

Ensemble 2 heads using pytorch.
- Backbone : ViT-H-14 + LAION2B
- Head1 : Dropout(0.1) -> Linear(64) -> Normalize
- Head2 : Linear(64) -> Normalize
- Output : Head1 + Head2 (add)

ViT -> Head1 / Head2 -> Normalize -> Ensemble

In training, backbone was freezed and trained head module  using arcface.

# Dataset

- Head1 : GLR2021 + ImageNet1K + Products10K
- Head2 : GLR2021 + ImageNet1K + Products10K + AlibabaGoods + FoodRecognition2022 + MET

# What did not work

some trial was by tensorflow implementation.

- ArcFace (large s + small margin)
- SubCenter ArcFace (Some K values)
- AdaCos
- Head : SelfAttention + Linear
- Head : Linear, ReLU, Dropout, Linear
- Head : Linear, ReLU, Dropout, Linear, BatchNormalization
- Contrastive Learning (SimCLR, SupCon)
- Focal Loss
- Single model or ensemble other CLIP(RN50, ViT-B, ViT-L (224px, 336px), ViT-g), CNN, other Head based arcface
- Ensemble method (concat + AveragePooling, weighted add)
- Using other dataset 

# Question

- DataAugmentation is not good?
- Ensemble of concatenating channel axis looked good, but score was lower than simple 64 dim
   model1 -> 32dim, model2 -> 32dim, ensemble (64 dim)
- Instance level label is not good?
- 2 stage classification is good?  1 stage : genre classsification , 2 stage : embedding  (I had no time to try)
