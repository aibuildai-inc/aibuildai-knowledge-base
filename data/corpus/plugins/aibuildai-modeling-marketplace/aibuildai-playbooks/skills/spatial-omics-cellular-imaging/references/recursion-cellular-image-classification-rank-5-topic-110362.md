# 5th place solution. AdaBN[domain==plate]

Competition: recursion-cellular-image-classification
Rank: #5
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110362

First, I would like to thank Recursion and Kaggle for organizing this interesting challenge, and thank team Double strand for their contribution.


Here's a summary of my solution.


### In this competition we have a multi-source and multi-target domain dataset. And we have domain labels.

### What is "domain" in this dataset?

cell, experiment, or plate

My EDA suggests \[domain==experiment\]. But \[domain==plate\] worked better in practice. This is reasonable since batch effects and plate effects exists.

### How to use domain labels? Simply [AdaBN](https://arxiv.org/abs/1603.04779).
In short, do not challenge your model(with BN layers) with cross-domain batches.

In training, use domain(plate) aware batch sampling.

In testing, use domain batch statistics in BN layers.

With batch norm done right, this competition **IMMEDIATELY** becomes a regular classification challenge: training converges smoothly and I had consistent validation/LB scores (except for HUVEC-05).

Some results of my early experiments, ResNet50, 224x224 input, same hyperparameters
- random batch sampling, val acc 40+%
- sample batches in the same cell type, val acc 50+%
- sample batches in the same experiment, val acc 60+%
- sample batches in the same plate, val acc 70+%

### Model
Sequential(BatchNorm2d(6), backbone, neck, head)

backbone: DenseNet201, ResNeXt101_32x8d, HRNet-W18, HRNet-W30

neck: gap or gap+bn

head: 5 fc layers (1 shared and 4 for different cells)

### Loss:

ArcFaceLoss(s=64, m=0.5) for gap+bn neck

ArcFaceLoss(s=64, m=0.3) for gap neck

### Exemplar Memory
In this dataset, we could get more supervision than siRNA labels.



[Exemplar Memory](https://arxiv.org/abs/1904.01990) fits this structure perfectly.

Fine tuning with 19 exemplar memory modules (HUVEC-05 and 18 test experiments) gave me ~1% LB boost, and HUVEC-05 validation accuracy increased from ~35% to ~46%(~65% with 277 linear assignment)

### Training
1108-way classifier with treatment only.
input 512 -&gt; random crop 384 -&gt; random rot90 -&gt; random hflip
loss = 0.5 * loss\_fc\_cell + 0.5 * loss\_fc\_shared

No pseudo labeling was used.

### Prediction
input 512
Use fc_cell.
No TTA.
Averaging two sites.
lapjv for linear assignment.

### DenseNet201 results

|  | Public | Private | Public (leak) | Private (leak) |
| --- | --- | --- | --- | --- |
| train data only | 0.92620 |0.97325 | 0.98307 | 0.99303 
| + exemplar memory fine tune | 0.95531 | 0.98321 | 0.98691 | 0.99394

### Some interesting finding

HUVEC-05 prediction of fc\_shared is always better than fc\_cell. What's wrong with this experiment?
