# 4th place solution and experience sharing

Competition: landmark-retrieval-2020
Rank: #4
Source: https://www.kaggle.com/c/landmark-retrieval-2020/discussion/175378

Hi all, here's our brief solution writeup:

#### What we have tried and works
* Augmentation: such as random crop and rotate. Using AutoAugmentation really helps.
* Backbones: Resnest200 and Resnet152 due to the resource limit.
* Pretrain: ImageNet pretrained and Softmax pretrained helps convergence.
* Loss Function: angular based loss function such as ArcFace.
* Label Smoothing
* Cosine learning rate with warmup
* Larger input size. We try 224, 336, 448 and 560. We choose 448 as final input size because of its higher cost performance. Smaller input size causes a drop in score.

#### What we have tried but not works
* EfficientNet B7: we trained b7 in pytorch and transferred it in TF savedmodel format, but it failed with **Notebook Timeout**.
* Some other large backbones but with lower scores: SEResNext, APolyNet, FishNet, HRNet.
* Some other hyper param in loss function such as larger or smaller margin.
* AdaBN
* DCN

#### What we haven't tried
* Larger backbones, such as Resnest269. 
* It seems that B7 works [here](https://www.kaggle.com/c/landmark-retrieval-2020/discussion/175306). Maybe the way we transfer our models from pytorch to tensorflow causes high time cost in the submission.
* Multi scale input like what baseline model has done.
* EMA
* KD

Trained models will be upload in a few days. Thanks.
