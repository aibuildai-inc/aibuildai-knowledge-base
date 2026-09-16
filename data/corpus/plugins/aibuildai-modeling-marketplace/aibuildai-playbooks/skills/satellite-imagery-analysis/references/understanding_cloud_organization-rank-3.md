# Finally GM & 1st time won prize money! And 3rd place solution.

Competition: understanding_cloud_organization
Rank: #3
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/117949

UPDATE: code is now available [here](https://github.com/naivelamb/kaggle-cloud-organization).

Thanks for Max Planck Institute for Meteorology and Kaggle for hosting such an interesting competition. Congrats to all the winners.

The key in my solution is training two segmentation models: **seg1** trained on all data with BCE loss, and **seg2** trained on non-empty images only with soft DICE loss. I think it works because this competition basically has two tasks: 1) detect the empty images; 2) predict accurate masks for the non-empty images. The two segmentation models address these two tasks respectively. 
## How I come up with this. 
I started the competition with resnet34-FPN using BCE loss (**seg1**). This model achieves ~0.608 on LB and the major contribution comes from capturing the empty mask: it captures ~80% empty masks. I tried a lot to improve the non-empty part, like using combo loss of BCE and DICE, but it is hard to improve the neg-dice (dice score for the empty masks) and pos-dice (dice score for the non-empty makes) simultaneously.

To predict the non-empty mask accurately, I decided to train 4 individual segmentation models for the non-empty images and then ensemble them together. Since all the train images are non-empty, we can use soft DICE loss directly and the model would focus on predicting accurate masks. I used exactly the same network structure, resnet34-FPN (**seg2**). Then I simply replace all the non-empty predictions from **seg1** model using the predictions from ‘seg2’. Only 1 fold of this 2-stage segmentation pipeline, no TTA, no min-size remover, no classifier, no threshold adjustment (all 0.5) could achieve LB 0.652. After including a resnet34 classifier (0.5 threshold), I got LB 0.655. 

Later on, I managed to train all 4 classes in one model by implementing pos-only soft DICE loss. The code looks like:

```python
def dice_only_pos(logits, labels, labels_fc):
    # logits -&gt; pixel level predictions
    # labels -&gt; pixel level labels
    # labels_fc -&gt; image/channel level labels
    pos_idx = (labels_fc &gt; 0.5)
    neg_idx = (labels_fc &lt; 0.5)
    loss = SoftDiceLoss()(logits[pos_idx], labels[pos_idx])
    return loss
```
This loss only counts the non-empty channels and ignores all the empty channels.

In summary the pipeline looks like: 
&gt;1. **seg1**: a multi-label segmentation model trained with BCE loss
&gt;2. **seg2**: a multi-label segmentation model trained with pos-only soft DICE loss
&gt;3. **cls**: a multi-label classifier trained with BCE loss. 

The final submission is achieved by the following steps:
&gt;1. Get predictions using **seg1**
&gt;2. Replacing the non-empty masks from **seg1** by predictions from **seg2**
&gt;3. Removing more empty masks using **cls**

Both pixel-level (segmentation) and image-level (classifier) thresholds are 0.5. 

## Baseline results for the 2-stage segmentation
Model summary:
&gt;Network: Resnet34-FPN 
&gt;Image size: 384x576
&gt;Batch size: 16
&gt;Optimizer: Adam
&gt;Scheduler: reduceLR for seg1, warmRestart for seg2.
&gt;Augmentations: H/V flip,  ShiftScalerRotate and GridDistortion
&gt;TTA: raw, Horizontal Flip, Vertical Flip

Results:
&gt;1-fold: 0.664 
&gt;5-fold + TTA3: 0.669
&gt;5-fold + TTA3 + classifier: 0.670. 

*TTA1 means only raw images; TTA3 means raw + H/V flip.*

The rest of my work is just trying different backbones to find the best one. My final models are:

&gt;seg1: densenet121-FPN, TTA1
&gt;seg2: b7-FPN, TTA3
&gt;cls: b1, TTA1

Results:
&gt;1-fold LB: 0.673
&gt;5-fold LB: 0.6788

## Ensemble

I ensembled multiple seg2 models using major vote. By including 4 models (b5-Unet, InceptionResnetV2-FPN, b7-FPN and b7-Unet), I achieved 0.6792 on LB. 

## Pseudo Labeling
I selected the pseudo labels based a LB 0.6790 submission with the following rules:
&gt;1. Empty channels with classifier prediction &lt; 0.3
&gt;2. Non-empty channels with classifier prediction &gt; 0.7

An image is selected when all the 4 channels satisfy one of the conditions. 835 images are selected. I retrained the b7-FPN and b1-classifier including the pseudo labeling samples, and the final models are:
&gt;seg1: densenet121-FPN, TTA1
&gt;seg2: b5-Unet + InceptionResnetV2-FPN + b7-Unet + b7-FPN + b7-FPN-PL, TTA3
&gt;cls: b1-PL, TTA3

*PL means the model is retrained with pseudo labels*

This model achieves 0.6794 LB. 

On the last day, I decide to optimize the classifier threshold channel wise to achieve the best local CV, which gives me 0.6805 LB. 

## Other things worth mentioning
1. My CV aligns pretty well with the LB. 1-fold CV = LB +- 0.005. 5-fold CV = LB - (0.010 ~ 0.012). This helps a lot during the model development.
2. Resizing the image before training could significantly reduce the training time. My resnet34-FPN could finish 1 epoch of training and validation in around 1 mins on a 2080Ti. 
3. For **seg1** and **cls**, complicated networks do not work. This is probably due to the noisy labels. For **seg2**, I cannot make seresnext50 and seresnext101 work and I have no idea why.
