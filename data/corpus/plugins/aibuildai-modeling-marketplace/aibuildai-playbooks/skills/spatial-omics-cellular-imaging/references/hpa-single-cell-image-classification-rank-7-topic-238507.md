# 7th place solution

Competition: hpa-single-cell-image-classification
Rank: #7
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238507

First of all, I would like to thank the host for organizing such an exciting competition. I learned a lot from the challenging tasks that I have never dealt with before, and I was helped by the many kind contributions to Discussion by the host. 

## Solution Overview

Our team gave up on improving the segmentation mask by HPA-Cell-Segmentation early on, and concentrated on improving the accuracy of multi-label classification for each cell.

We mainly experimented with following two approaches:
1. Predict each cell from a Class Activation Map (CAM) of image level classifiers, as is commonly used in Weakly Supervised Semantic Segmentation.
2. Crop the image for each cell and predict one by one

In approach 1, the image level label is cleaner (compared to using it as a cell level label), so it is easier to train the classifier. On the other hand, it has a disadvantage for images with high SCV because the prediction of individual cells is easily affected by neighboring cells. In approach 2, it is difficult to perform well by simply using the image level label. But it is less affected by SVC because prediction is done for each cell separately. To take advantage of these two complementary approaches, we created models with both approaches and used them as an ensemble.

The training pipeline is shown in the following image.

We repeated the offline pseudo learning process twice, training a new model using pseudo labels from an ensemble of multiple models and TTA. All pseudo labels are soft labels after applying the sigmoid function. All ensembles were done with simple average, and the TTA used all D4 augmentation. The image level classifier was trained using 768 x 768 images except for the 1536 one, and the cell level classifier was trained using 192 x 192 images. Cosine classifier uses cosine similarity between feature map and linear layer instead of linear transformation of feature map (We follow the equation 3 of https://arxiv.org/abs/2103.16370)

## Data
We used all the training data from this competition and the public HPA data.

## Validation Strategy
To split the data, we used MultilabelStratifiedKFold from [iterative-stratification](https://github.com/trent-b/iterative-stratification) with 5 folds. We mainly monitored image level mAP, Focal loss, and binary cross entropy, but we could not find any metrics that correlated with public LB, so we relied on feedback from public LB.

## Image Level Classifier


For the image level classifier, in addition to image level Focal loss, we used a consistency loss such that the prediction of the cell level under weak augmentation matches the prediction of the cell level under strong augmentation (CutMix). For cell level prediction, we used the average of the CAM in the region occupied by each cell (since the number of channels in CAMs is small, it worked reasonably fast even using such as scatter_add). The idea of the consistency loss is based on [PseudoSeg](https://arxiv.org/abs/2010.09713) and [PuzzleCAM](https://arxiv.org/abs/2101.11253) (I think the reconstruction loss in PuzzleCAM can be regarded as a consistency loss using a variant of Cutout).

We mainly used EfficientNet-B2 as the image level classifier. This is because using other architectures (We tried ResNet and ResNeSt) or the larger EfficientNet would have improved the local image level mAP, but not the public LB. (This choice may have caused the public LB to overfit).

When using a pseudo label from an ensemble of other models, "Cell Level Pseudo Label" in the figure is replaced with the pseudo label from the ensemble.

## Cell level Classifier


For the Cell level Classifier, in order to input both the shape of the entire cell and the size of the cell into the CNN at a somewhat small resolution, we concatenated both a fixed-scale, nucleus-centered crop and a variable-scale, whole-cell crop into the CNN. We did not use a model trained with a simple image level label because it did not perform well.

## Post Processing

From the following comment in the [single-cell-patterns notebook](https://www.kaggle.com/lnhtrang/single-cell-patterns#18.-Negative):

> Please also note that border cells where most of the cells are out of the field of view and to cells that have been damaged or suffer from staining artifacts. A good rule of thumb (that our annotators used in generating ground truth) is if more than half of the cell is not present, don't predict it!

We scaled the confidence with a value based on the area of its cell (shown as “edge scale” in the bellow image) so that the confidence of the small cells at the edges of the image would be small. We also scaled the confidence of cells that are not at the edge of the image by a value (shown as non-edge scale), assuming that smaller cells are harder to predict.

[image]


## Scores

|Training|Architecture|Pseudo Label|Public LB|Private LB|
| --- | --- | --- | --- | --- |
|Image level classification|EfficientNet-B2|-|0.531|-|
|Image level classification|EfficientNet-B5|-|0.526|-|
|Image level classification|EfficientNet-B7|-|0.502|-|
|Image level classification|EfficientNet-B2 (1536)|-|0.526|-|
|Image level classification|EfficientNet-B2|1st|0.554|-|
|Image level classification|EfficientNet-B2-cos|2nd|0.554|-|
|Image level classification|EfficientNet-B2-cos|2nd|0.566|-|
|Cell level classification|ResNeSt50|1st|0.551|-|
|Cell level classification|ResNeSt50|2nd|0.571|-|
|Cell level classification|ResNeSt50|2nd|0.569|-|
|-|Final Ensemble|-|0.580|-|
|-|Final Ensemble-postprocess|-|0.594|0.540|

## Code

(Added on May 28, 2021) We have published the code.
https://github.com/pfnet-research/kaggle-hpa-2021-7th-place-solution
