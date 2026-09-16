# 5th place solution write-up

Competition: landmark-retrieval-2020
Rank: #5
Source: https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176151

I would like to thank my teammate @aerdem4 for his work, discussions and ideas throughout this challenge, our first team-up turned out to be quite successful ✊. Congratulations to all the winning teams and solo winners ! Finally, many thanks to the organizers and Kaggle for hosting this interesting competition.
Our final submission consisted of 4 models from two different architectures (gempool cnn and delg). 

**1. Data pre-processing**
+ Training only on clean train set (81313 classes), validation by computing global average precision metric on a subset of 120000 images (having overlapping classes with train set) randomly sampled from the index set.
+ Data augmentations: resize image longer side to [544, 672] then random crop 512x512, RandAugment + Cutout.

**2. Modeling**
+ Two architectures: CNN (with SEResNeXt50, SEResNeXt101 and ResNeXt101-32x4d as backbones) with CosFace head, and DELG (re-implemented in PyTorch, with SEResNet101 as backbone).
+ Generalized mean pooling with frozen p set to 3 was used; bottleneck structure (GEMPool(2048) -> Linear(512) -> BatchNorm1d -> CosFace(81313)) to reduce computation. 
+ Models were trained either in 10 or 20 epochs with AdamW optimizer and warm-up cosine annealing scheduler.
+ Focal loss and label smoothing were better than cross entropy loss

**3. Inference**
+ Features (512-dim) extracted at scale 1 for each model.
+ Concatenate 4 models’ features into a 2048-dim vector.
+ Kernel runtime: 8 hour 20 minutes

**4. Public/Private performance**
|Methods  |Epochs  |Public  |Private  |
| --- | --- |
|resnext101 gem |20  | 0.34596 |0.31024  |
|seresnext50 gem  |20  |0.3349  |0.29811 |
|seresnext101 gem  |20  |0.34749  |0.31282  |
|seresnet101 delg  |10  |0.336  |0.29882  |
|ensemble  |  |0.36644  |0.32878  |

**5. Ablations on DELG vs GEM**
We used SEResNeXt50 for all experiments in this section. 
|Methods  |Epochs  |Public  |Private  |
| --- | --- |
|gem |10  | 0.32163 |0.28434  |
|gem + self attention block after res5  |15  |0.32006  |0.28187 |
|gem + online hard neg mining  |15  |0.32357  |0.28687  |
|gem + focal loss  |20  |0.32634  |0.29054  |
|delg + focal loss  |20  |0.32928  |0.29263  |
|gem + focal loss  + cutout |20  |0.3349  |0.29811  |

**6. Things that didn't work for us**
+ Pre-training on v1 dataset hurt.
+ Training on concatenated v1 and v2 data.
+ Earlier, I trained an EfficientNet B3 and found out that given the same training configs/ epochs, it performed worse than my baseline ResNet50 (which scored 0.299 on public LB). EfficientNets seem to only work well when you train them long enough with hard augmentations. After reading the 1st place solution, I know which experiment I'm gonna run next for the recognition challenge 😁
