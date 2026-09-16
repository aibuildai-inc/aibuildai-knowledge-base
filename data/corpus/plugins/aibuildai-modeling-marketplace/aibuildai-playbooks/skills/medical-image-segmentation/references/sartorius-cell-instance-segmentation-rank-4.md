# 4th place solution

Competition: sartorius-cell-instance-segmentation
Rank: #4
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298146

Thanks to Kaggle and Sartorius for this interesting competition. I also thank my teammates, @tanakar and @tereka, @ren4yu. I learned a lot from them.

Our experiments are based on CBNetV2 [repo](https://github.com/VDIGPKU/CBNetV2)

In the following, I want to give a summary of our solution.

# Overall
Our solution consists of three parts: classification part, instance segmentation part, post-processing part. Each cell type has different instance segmentation models and post-processing, so classification is necessary.
[[overall.png]](https://postimg.cc/jnNJBNts)

# Classification
3class CBNet DBS Cascade-RCNN  (num_classes=3) was used as a classification model. For a given image, this model classifies the image into the cell type with the highest number of detected cells. 
This model is first pre-trained with LIVE_CELL (num_classes=1) and then fine-tuned with the competition's train data (num_classes=3).
I tested this on semi-supervised data and it was able to classify them perfectly.

# Instance Segmentation
We prepared at least one model for each of shsy5y, astro, and cort. These models were trained in different settings and with different data. 
## Pseudo Labeling
@tereka 
For all cell types, pseudo labeling on semi-supervised data improved both CV and LB.
Because of the small number of train data, it was better to use three types of cells when using only train data. However, because of the large number of semi-supervised data, we were able to change the data used for each cell type.
[Pseudo labeling](https://drive.google.com/file/d/1PSIxdNiwtMwTV3Wq6AJbWcp2plnoIVQA/view?usp=sharing)
### Models for pseudo labeling
Data: live_cell → train (all cell types)
Model: 3 types of 1class CBNet DBS Cascade-RCNN
We changed the MMdet config file depending on the target cell type.

[[train-strategy.png]](https://postimg.cc/DW2dsCF1)
## shsy5y models
@tyaiga 
Data1: live_cell → train (all cell types)
Model1: CBNet DBS Cascade-RCNN
Data2: live_cell → semi-sup+train (shsy5y & cort) → train (shsy5y)
Model2: CBNet DBS Cascade-RCNN
The number of cells in shsy5y and cort are very different, but the individual cells are similar, so, we used the two as training data when using semi-sup data.

## astro models
@tyaiga 
Data: live_cell → semi-sup+train (astro) → train (astro)
Since astro has a very different shape and size from the other cells, we improved the score by using only astro data fot train data.
In astro, I only used one model because the ensemble did not work well.

## cort models
@tanakar 
Data: live_cell → semi-sup+train (shsy5y & cort) → train (court)
Model: 2 x HTC resnext64x4d, 2 x CBNet DBS Cascade-RCNN
These four models were combined into one model using the method below.

### ensemble detection model
@tereka @ren4yu 
We only used the two-stage model. So, we used an ensemble method like [this](https://github.com/amirassov/kaggle-imaterialist), where RPNs are connected and treated like a single two-stage model.
This improved cort score.
[[cort-models.png]](https://postimg.cc/pmvDb0m3)

## Augmentation
Multiscale: astro and cort → (1280, 1280)~(1792, 1792), shsy5y → (1280, 1280)~(1536, 1536)
Horizontal Flip
Vertical Flip

## TTA
Multiscale + horizontal flip (vertical flip and diagonal flip didn't work)

# Post-Processing

## astro pp
@ren4yu 
The astro annotation was broken, so I reproduced it with cv2.findContours and cv2.fillConvexPoly.

## fix-overlap
@tyaiga 
In this competition, predictions are not allowed to overlap. So, we have to eliminate overlap part in post-processing. 
In our fix-overlap process, first we took cells with a higher cofidence score than classwise threshold. Then, we processed from the highest score to the lowest, and deleted instances with a large percentage of already-used area. In addition, only shsy5y score was improved by removing those with pixels lower than threshold.

## semantic re-lank
@ren4yu 
As shown in the Refinemask [paper](https://arxiv.org/abs/2104.08569), the confidence score of the instance segmentation model does not reflect the correctness of the mask. Therefore, we modified the scores and re-lank the instances with semantic segmentation model (UNet++).

## fix-overlap ensemble
@tyaiga @tanakar @ren4yu 
In this ensemble method, we first concat the output of multiple models and sort them by their semantic relank scores. Next, we applied fix-overlap pp to remove the overlap. This improved shsy5y score.
[[fixoverlap-ensemble.png]](https://postimg.cc/FfRkNwNC)

## WBF with mask
@tereka 
We tried WBF extended for mask. This improved Public LB score, but 'ensemble detection model' is better for Private LB.

# Tips
The default config file of MMDetection is fitted for COCO. Depending on the shape of the instances and the number of instances in an image, it is necessary to change the settings for training.
e.g. anchor_generator.ratios, rpn_proposal.nms_pre, and so on...
