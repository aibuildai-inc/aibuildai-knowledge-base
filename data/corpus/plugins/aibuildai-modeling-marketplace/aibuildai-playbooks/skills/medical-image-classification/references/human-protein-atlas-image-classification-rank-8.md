# 8 place solution writeup

Competition: human-protein-atlas-image-classification
Rank: #8
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77251

First of all, thanks to Kaggle and HPA team for this interesting competition! Even leak couldn’t spoil it! 

Our solution is an ensemble of 14 models. Most of them were trained on 512x512 RGB with additional data.

Many thanks Jeremy Howard for great fast.ai framework! It’s nice how easy it is to patch it. Fast.ai has been changed 40 times since the competition began, so it’d have hurt if it was not open sourced.

Our models are: Se-ResNext-50 trained on 256x256, 512x512 and 768x768 sizes, InceptionV4, BN-Inception and Xception (all trained on 512x512).  
We didn’t have enough resources to train models on high resolution images. There were two ways to deal with it: resizing and crops, but training on crops is risky since some organelles occur only once per image (e.g. cytokinetic bridge), and it is difficult to select proper crops, so we didn’t do it.

Things that worked:
1. Learning rate finder and cyclic learning rate (long cycles at the beginning, short cycles at the end).
2. Differential learning rate with gradual reducing (as described here: https://blog.slavv.com/differential-learning-rates-59eff5209a4f ) helped to preserve weights from ImageNet.
3. Focal loss with default gamma, LSEP loss ( https://arxiv.org/pdf/1704.03135.pdf ).
4. Simple one-layer network head.
5. Brightness augmentations, D4 and wrap transforms.
6. Average of 32 TTAs (we used same augmentations as during training).

Things that didn’t work:
1. Training on RGBY.
2. Training with sample pairing ( https://arxiv.org/abs/1801.02929 ).
3. Mixup, which probably didn’t work because green channel is too important ( https://arxiv.org/abs/1710.09412 ).
4. Complex network head.
5. Large architectures such as Nasnet or Senet-154 (they would have probably worked, if we had more GPUs).
6. Training one-vs-all models and training on subsets of similar classes.
7. Training classifier over bottleneck features of networks (we tried lots of approaches here, but unfortunately all of them proved to be worse than our models).
8. Complex augmentations, such as green channel modifications discussed here: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/75768

External data was acquired from Human Protein Atlas site. After trying to find duplicates between train and externaldata we found out that labels wasn't match some times. First of all they shouldn't be taken from subcellular_location.tsv as these locations is raleted to gene/protein not to the sample. But parsing labels from xml files also wasn't quite correct. More correct labels are actually on web-site, so we took them. And it looks like they merged some rare labels that wasn't presented in our task into others, so we do the same.

Our resources: 4x1080ti. It seems that the full training cycle consumed approximately two weeks of computing time.

Findings:
1. Yellow channel wasn’t very helpfulas many participants noticed on the forum.
2. Labeling was probably quite noisy (we found several contradictions between HPAv18 data and data provided in this competition).
3. Several classes are extremely similar visually. It is almost impossible to distinguish between lysosomes and endosomes, for example (no wonder: endosomes are literally reborn into lysosomes at some point of their lifecycle). So it is not surprising that models don’t perform well enough on these classes too.

Our validation is a kind of an Adversarial Validation. We arranged train by similarity to the test using an NN with simple architecture and got 8K samples as a holdout. We used this holdout to fit the thresholds for single models and check the scores. We tried our best to avoid duplicates between train and our holdout. All leaked images from HPA were added to the validation set too.

The most challenging part of the competition is how to deal with “small” classes with only a few positive cases. We couldn’t handle it better than ensembling models with linear models and voting between those stacks. 

Other classes were stacked with LightGBM model. It was stable enough on the validation set because we followed the “folds-inside-folds” scheme (similar to Strategy C from https://www.kaggle.com/general/18793 ). The gap between the local validation and leader board was stable with 0.01 precision. It wasn’t as accurate as we wanted it to be, but still it was ok to trust our validation.

P.S. Some stats:
Our group chat contains 1325 screenshots, 321 files, 428 links and thousands of messages.
