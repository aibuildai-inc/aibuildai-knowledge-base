# 33th Place Algorithm on Private LB

Competition: human-protein-atlas-image-classification
Rank: #33
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77256

I participate this competition as a kind of test. I wanted to develop a algorithm, which generate reasonable models without lots of human intervention.

Since data preprocessing and augmentations are very labor intensive works, I focused on cross-validated, various-architecture, robust single models and ensembled models from them, with very clean and basic script.

My single models are like 0.55~0.57 on public LB. Unfortunately, unlike my expectations, ensemble from LOTS of models wasn't that much effective(0.601 on public LB). I learned that basic single model should perform better than this. If I have a more time(I participated late), I will definately try to improve my single models, with changing input size, data augmentations, various network architecture, ROI cropping and ensemble with other known hand crafted features for this dataset.

Here is the github repository so I hope this is helpful for many people. 
https://github.com/ildoonet/kaggle-human-protein-atlas-image-classification

### Models

- vgg16
- resnet50, resnet101, ...
- densenet121, densenet169 *
- inception v3, inception v4 *
- se152
- polynet
- NASNet, PNASNet

### Implementations

- Data Loader for External Datas and Merger
- Basic data augmentations
  - Rotation, Flip *
  - Channel drops
- 16 Test-Time Augmentation
- 5-Folds Cross Validation
- Simple Threshold Search Algorithm
- Ensembles
  - Test-Time Augmentation Averaging *
  - Majority Voting *
  - Fully-Connected Neural Network
    - logits -&gt; output
    - logits + features -&gt; output
  - XGBoost
- Loss
  - Soft F1 Loss *
  - Binary Cross Entropy *
  - Focal Loss
  - MultiLabelMarginLoss
