# 4th Place Solution: Stabilizing Convergence in Understanding Clouds

Competition: understanding_cloud_organization
Rank: #4
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118016



First of all, I would like to express my gratitude and appreciation to the following parties for organizing such a great competition:
 - [Kaggle](https://www.kaggle.com)
 - [Max Planck Institute for Meteorology] (https://www.kaggle.com/MaxPlanckInstitute)

Besides, I would like to use this opportunity to thank my fellow kagglers for all the insightful posts in the discussion forum of various competitions. I have also learned a lot of stuffs and gained knowledge by reading from past solutions. There is a good thriving culture of idea sharing and contributions which I have found in every corner of Kaggle and I loved to be part of it.

## The Main Challenge
The challenge that I have faced initially in this competition is that many models of different architectures tend to overfit easily in early training stage especially for the larger and deeper models, such as SE-ResNext-101 and EfficientNet B5-B7. I have suspected the culprit might be due to the labels given are too noisy and this increases the tendency of model to be overfitted to the noises of training data, as the labels were determined by the union of the areas marked by all annotators. Also, the shape of the label provided is rectangular instead of the exact shape fitted to the boundary of cloud patterns. I understand the reasons behind [these decisions made by the competition host](https://arxiv.org/pdf/1906.01906.pdf), and here goes my whole journey of this competition, which is revolved around stabilizing the convergence in training models.
 
## Solution Overview
My solution for this competition is mainly comprised of the followings:

- **Pure segmentation models without false positive classifier**
After reaching public LB 0.6752 with segmentation model, I've trained a few classifiers using Resnet34, SE-ResNext-50 and EfficientNet-B4 but the performance is pretty unstable (+/- 0.003 ~ 0.010) in local cross validations of 10 folds. Thus, I discarded the classifiers and decided to stick with segmentation models.

- **Network Architectures**
I've used the awesome implementations of various models from [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch), [pretrained-models.pytorch
](https://github.com/Cadene/pretrained-models.pytorch), [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) and [Resnet34-ASPP](https://www.kaggle.com/c/understanding_cloud_organization/discussion/115787#671393) from @hengck23. My final ensemble used 7 folds of EfficientNet-B4-FPN and 3 folds of Resnet34-ASPP as they have better performance and more stable in error convergence in my case after running rounds of experiments using various network architectures.
- **RAdam Optimizer**
RAdam helped to stabilize training error convergence as it is less sensitive to learning rate change in my case, thus minimizing the variance.
- **Flat threshold of 0.4 for all classes**
Threshold of 0.4 yielded the highest cross validation DICE score when compared in the range of [0.4, 0.5, 0.6], no further fine-tuning of threshold is done.
- **Minimum segmentation mask size of 5000 pixels for all classes**
The mask size threshold is set to be just high enough to filter out noises, no any other post-processing methods is used.
- **Input Size**
Downsized from the raw size of 1400 x 2100 to 700 x 1050. After applying augmentations, it is downsized again from 700 x 1050 to 384 x 576.
- **Augmentations used in training**
    - horizontal flip
    - vertical flip
    - random shift, scale and rotate
- **Test-time Augmentations (TTA)**:
  - horizontal flip
  - vertical flip
  - 180 degree flip (horizontal + vertical flip)

- **Pseudo-labeling**
 I've used two approach for pseudo-labeling, one in which only the confident pseudo-labels are selected and use in training, another in which pseudo-labels are generated from all the test data. In my case, the model training performance of using pseudo-labels from all test data is more robust and stable in terms of error convergence and achieve higher DICE score.
- **Ensemble with equal weight averaging**  
- **Trained initially with BCE Loss, fine-tuned with Symmetric Lovasz Loss originated from this [paper](https://arxiv.org/abs/1705.08790) and modified by @tugstugi**
Below is the PyTorch implementation code of Symmetric Lovasz Loss:
```
def symmetric_lovasz_loss(outputs, targets):
    batch_size, num_class, H, W = outputs.shape
    outputs = outputs.contiguous().view(-1, H, W)
    targets = targets.contiguous().view(-1, H, W)
    return (lovasz_hinge(outputs, targets) 
      + lovasz_hinge(-outputs, 1 - targets))/2
```
- **GPU used**
  - 2 x RTX2080Ti

## Conclusion
I think local cross validation is very important and we should always believe in it despite the score showed on Public LB might be lower or higher as it is only computed based on a minor subset of the test dataset. Besides, the **combination of RAdam optimizer, Symmetric Lovasz Loss, Pseudo-labeling and ensembling** has helped significantly in stabilizing the convergence and improving the score.
<br><br>
Thanks for reading! See you again in upcoming competitions.
