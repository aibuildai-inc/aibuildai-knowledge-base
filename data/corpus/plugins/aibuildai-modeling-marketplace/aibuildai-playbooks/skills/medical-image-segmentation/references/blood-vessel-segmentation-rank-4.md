# 4th place solution. Boundary DoU Loss is all you need!

Competition: blood-vessel-segmentation
Rank: #4
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475052

First of all, I would like to start my solution description with a few important words:

*I would like to thank the Armed Forces of Ukraine, the Security Service of Ukraine, Defence Intelligence of Ukraine, and the State Emergency Service of Ukraine for providing safety and security to participate in this great competition, complete this work, and help science, technology, and business not to stop but to move forward.*

# Context
- Business context: https://www.kaggle.com/competitions/blood-vessel-segmentation 
- Data context: https://www.kaggle.com/competitions/blood-vessel-segmentation/data

# Overview of the approach:
My final model is a mixture of 2d and 3d models with d4 tta. For the 2d model, the multiview tta was applied. All models were trained in a 2-fold setup with kidney_2 and kidney_3_dense selected as validation sets. The ensembling was performed with equal weights for both 2d and 3d models.



# Details of the submission

## Data preparation and training data and validation scheme 

All final (3d and 2d) models were trained on kidney_1_dense, kidney_2, kidney_3_dense, kidney_3_sparse and pseudo labels [50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_](http://human-organ-atlas.esrf.eu). Initially, I used slice-wise normalization to normalize images but later switched to stack-wise normalization based on percentiles.

The 2D model was trained in a multiview setup: all images were stacked in a tensor and sliced in different axes afterward. During the training, the set of augmentations and sampling strategy was crucial. The weighted sampling was based on sparsity percentage: dense samples had a weight of 1, while sparse samples had a weight equal to their sparsity. For pseudo labels,  I chose the same weight as for kidney_2, e.g.: 

```python
kidney_1_dense: 1, 
kidney_2: 0.65, 
kidney_3_dense: 1, 
kidney_3_sparse: 0.85, 
50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_: 0.65. 
```

The augmentation scheme was the next one, with a chance of 0.5 CutMix augmentation being applied. The cropping was performed from the same organ and the same projection axis. Afterward, on top of CutMix, the next augmentation pipeline was applied:

```python
A.Compose(
    [
          A.PadIfNeeded(*crop_size),
          A.CropNonEmptyMaskIfExists(*crop_size, p=1.0),
          A.ShiftScaleRotate(scale_limit=0.2),
          A.HorizontalFlip(p=0.5),
          A.VerticalFlip(p=0.5),
          A.RandomRotate90(p=0.5),
          A.OneOf([
                A.RandomBrightnessContrast(), 
                A.RandomBrightness(), 
                A.RandomGamma(),
          ],p=1.0,),
    ],p=1.0,)
```

The crop size was set to 512. I’ve also tried higher resolution, but it performs +- the same result. 

I did some experiments with 2.5d approaches (3 and 5 channels), but it produced the same result or worse. 

The 3d model augmentation scheme contained only d4 augmentations and random crops. The cropping was performed with a 0.5 probability of an empty mask. This was motivated by false positives that appeared outside the kidney volume. This could be improved by incorporating the two-class 3D segmentation, but I didn’t have much time and resources to perform such an experiment. Thus, I decided to create a post-processing that would handle this. 
The crop size for the 3d model was 192x192x192.

Both models were trained in a 2-fold setup where as validation, I used kidney_2 (fold_1) and kidney_3_dense (fold_0). Removal of kidney_1 from the training set caused performance degradation in performance in both CV and LB, so I dropped the fold_2 and didn't perform training in that setup.

## Model setup
The best results I was able to get using the efficientnet family models with UnetPlusPlus decoder and SCSE attention from the segmentation_models_pytorch library. I’ve tried the resnet50 model, like it was mentioned in the discussion section, different transformers and seresnext models, but could overcome the performance of efficientnet-b5 (which performed the best on both CV and LB). On my local validation, the score I was able to get with efficientnet_b7 encoder and mit_b5 encoder, but on the LB the score was significantly lower.
The training was performed for 30 epochs with a Cosine LR scheduler starting from 3e-4 to 1e-6. I saved the top 3 checkpoints and used the best-last checkpoint for the submission.
The model efficientnet_b5_UnetPlusPlus trained in such a setup was able to score 0.878 on the public LB and 0.714 on the private LB at a 0.05 threshold. 

Here yellow is TP, green is FP, and red is FN.



The 3d model was heavily inspired by the nnUnet model architecture and was pretty much the same. Instead of the native nnUnet model, I used DynUnet from the monai library with almost default configuration and trained in almost the same setup as for nnUnet. As the optimizer, I used SGD with initial LR 0.01 and the Cosine Annealing LR scheme instead of LinearLR and trained for 500 epochs with 2000 samples per epoch. 
This model scored 0.869 (0.868 and 0.866 -- 0 and 1 folds respectively) on the public LB and 0.694 on the private LB (0.758 and 0.663 -- 0 and 1 folds respectively). 

Both models were trained using the BoundaryDOULoss (https://arxiv.org/pdf/2308.00220.pdf), which performed the best. I’ve tried to modify it to perform better on sparse data but failed. 

## Pseudo labeling 

Based on the preprint, I downloaded the additional data from http://human-organ-atlas.esrf.eu site (2 datasets). It appeared, that one of the datasets overlaps with the kidney_3, so I dropped it to prevent leakage. I used the other one to generate pseudo labels. For pseudo labeling, I used an ensemble of 2d models (efficientnet-b5 and efficientnet-b6 with UnetPlusPlus) trained with the same setup but without CutMix. The correct setup of CutMix as well as the 3d model I was able to discover close to the competition deadline, so I didn’t retrain the original ensemble and stick to the first version of pseudos. 

## Inference setup and Post-processing 

The inference for both models was performed using sliding_window_inference from monai library. Additionally, for 2d model I performed multi-view tta, which helped to detect small vessels and improve overall performance. 

For the 2d model, the crop size was 800 pix, while for the 3d – 256 pix with 0.25 overlap and Gaussian merging. All models used d4_transform from ttach library. I’ve forked the ttach repository and implemented the logic for 3d images, but the inference time increased significantly, and there was no major boost in performance, so I’ve sticked with 2d d4_transform for both 2d and 3d models :)

As I mentioned before, the 3d model had decent performance on the non-empty cubes, while empty ones were confusing the model. To handle this issue, decided to experiment with post-processing. The idea was the next one: let's try to find ROI where the vessels were presented. Since the 2d model didn’t have such a problem I’ve decided to find a bounding polygon for vessels for each 2d slice. Having a mask of ROI, I multiplied it with 3d model predictions and got a boost from 0.869 to 0.881 public LB and 0.701 private LB for a single 3d model.

Ensembling the 2d model and 3d model predictions with weights 1 and 1, I was able to improve the score from 0.881 to 0.884 on the public LB and 0.712 on the private LB.

Another post-processing approach that I’ve tried is to use Canny filters from cv2 to segment the kidney. This segmentation algorithm was not perfect, but applying such post-processing boosted my score from 0.884 to 0.892 on the public LB while failing on the private LB, scoring just 0.313.

## What didn’t work
- nnUnet out of the box. At the beginning of the challenge and after the pre-print reading, I tried to reproduce the result with nnUnet. The local score was promising, but the LB was 0. My intuition behind this issue related to the data normalization and spacing (scale), but I didn’t try to fix it and decided to build my own solution.
- BCE and Focal Loss.
- Transformers in both 2d and 3d model
- Zoom and brightness augmentation for 3d images
- Pseudo on top of sparse datasets. I’ve tried to fulfill the sparsity of the dataset by pseudo labeling and aggregation, but it didn’t improve the score.
- Additional projections. I’ve performed experiments with 2d models and additional slices generated from the 3d stack, but LB performance dropped by 20% while CV was about the same. 
- Auxiliary outputs such as distance transform or center of mass. 
- **and the most important: validation**


P.S. If you were able to read all of this, the top score on the private LB was a simple mix of 2d and 2.5d models with 1 and 3 channels :) 

P.P.S. Thank you for reading!

## Links
- [Pseudo labels ](https://www.kaggle.com/datasets/igorkrashenyi/50um-ladaf-2020-31-kidney-pag-0-01-0-02-jp2 )
- [Source code](https://github.com/burnmyletters/blood-vessel-segmentation-public)
- Inference code https://www.kaggle.com/code/igorkrashenyi/4th-place-solution/notebook + https://www.kaggle.com/code/igorkrashenyi/fork-of-multiview-2-5-sennet-hoa-inference-v3
