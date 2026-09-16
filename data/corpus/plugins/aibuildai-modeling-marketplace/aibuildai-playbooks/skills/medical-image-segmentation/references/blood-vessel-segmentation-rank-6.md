# 6th place solution: Luck is All You Need

Competition: blood-vessel-segmentation
Rank: #6
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475252

Hi, Kagglers!

After finding yourself on the Leaderboard after Grand Shakeup and restoring your mental health, we can dive deep into 6th place solution, but before this, a few very important words:
*I would like to thank the Armed Forces of Ukraine, Security Service of Ukraine, Defence Intelligence of Ukraine, and the State Emergency Service of Ukraine for providing safety and security to participate in this great competition, complete this work, and help science, technology, and business not to stop but to move forward.*

# Validation. Not really 

I was using for validation 2 organs (in different folds): kidney_3_dense and kidney_2. After releasing a [fast version of 3D Surface Dice](https://www.kaggle.com/code/junkoda/fast-surface-dice-computation), I was able to compute validation scores while training, and I received the next insights:
- Tracking the score on kidney_2 was useless for me. The validation score decreased from epochs 2-3 on kidney_2
- Scores on kidney_3_dense were meaningful for checking “radical” features, like additional data and new losses. But then optimal score fluctuated between 0.9-0.925 dice without any reasonable correlation with Public or Private score
- The optimal threshold on kidney_3_dense was optimal for Private, Public, and kidney_3_dense scores - 0.1 and lower 

- Resize to constant um/voxel (I have picked 50 um/voxel) for prediction increased optimal threshold both on CV and Public but decreased optimal score dramatically. But on Private, it became one of the most robust approaches

In summary, Validation did not work (at least mine). It is not strange because of solo data point in CV, Public, and Private 

# Data 

I was using all train data except kidney_1_voi sample
In order to enlarge training data I have used 50um_LADAF_2020_31_kidney_pag from [Human Organ Atlas](https://human-organ-atlas.esrf.eu/search?organ=kidney) 
For data normalization, I was using the approach proposed by @hengck23 - [percentile normalization](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/456118#2552053)

# Training setup

I mostly stick to the 2.5D approach with 5 slices.
I started from one view model and iterated along the last axis, but then I switched to a multiview and used slices by all three axes in the training
I have used 512 square crops with Non Empty probability of 0.5, pretty much standard augmentations and CutMix within one organ and one view with 0.5 probability and 1.0 alpha :
```python
"cutmix_transform":lambda : [
    A.PadIfNeeded(
        min_height=crop_size,
        min_width=crop_size,
        always_apply=True,
    ),
    # Sample Non-Empty mask with prob 0.5
    # Otherwise empty OR Non-Empty mask will be sampled
    A.OneOrOther(
        first=A.CropNonEmptyMaskIfExists(crop_size, crop_size), 
        second=A.RandomCrop(crop_size, crop_size), 
        p=0.5
    ),
],
"transform": A.Compose(
    [
        A.ShiftScaleRotate(
            scale_limit=0.2,
        ),
        # dihedral_aug
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.RandomBrightness(),
                A.RandomGamma(),
            ],
            p=1.0,
        ),
        ToTensorV2(transpose_mask=True),
    ]
),

"do_cutmix": True,
"cutmix_params": {"prob": 0.5, "alpha": 1.0},
```
I was using Adam optimizer and reduced learning rate with CosineAnnealingLR starting from 1e-3 and ending with 1e-6
Regarding loss function choice, I started with classical BCE+Dice loss and then tried to implement the loss function, which will directly optimize the metric, but unfortunately, it did not work well. Luckily, I have come across BoundaryLoss, which worked firstly comparable to BCE+Dice loss and then better. Interesting fact that the best (not selected) model was trained on BoundaryLoss + 0.5 Focal Symmetric Loss and scored 0.756 on Private, 0.867 on Public, and 0.916 on kidney_3_dense, which is a pretty much balanced score (of course, comparing to all other models score distribution 🙂)
I was training for 30 epochs. repeating the original train set 3 times and the pseudo train set 2 times
I have used a batch size of 14 samples and trained with DDP on 2 GPUs, so the final batch size was 28

After I saw [the post](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/461213#2594751) about promising results from 3D models, I started exploring 3D approaches, and they worked pretty well. In order to make it train without NaNs I have changed the optimization strategy and switched to SGD, with momentum=0.99, weight_decay=3e-5, nesterov=True and also changed starting learning rate to 1e-6 - taken from [monai example](https://github.com/Project-MONAI/tutorials/tree/main/modules/dynunet_pipeline)
As the overall image resolution of the image was increased dramatically, I had to reduce the batch to 3 on one GPU, so the aggregated batch size was 6. I was training in total for ~120K iterations
As for augmentations - they were pretty much the same as in the 2.5D setup, except from Zoom. 
```python
"transform_init": lambda : mt.Compose(
    [
        mt.OneOf([
            mt.RandRotate90d(keys=('image', 'mask'), prob=0.5, spatial_axes=(-3,-1)),
            mt.RandRotate90d(keys=('image', 'mask'), prob=0.5, spatial_axes=(-2,-1)),
            mt.RandRotate90d(keys=('image', 'mask'), prob=0.5, spatial_axes=(-3,-2))
        ]),
        mt.RandFlipd(keys=('image', 'mask'), prob=0.5, spatial_axis=-1),
        mt.RandFlipd(keys=('image', 'mask'), prob=0.5, spatial_axis=-2),
        mt.RandFlipd(keys=('image', 'mask'), prob=0.5, spatial_axis=-3),

        mt.RandScaleIntensityd(keys=('image'), prob=0.5, factors=0.2)
    ]
),
```
Zoom worked better on CV but worse on Public and also on Private (Why? - who knows …)

# Neural Networks 

I was mostly using [EfficientNet family](https://arxiv.org/abs/1905.11946) as an Encoder (from noisy student weights), started from B3, then switched to B5, and unfortunately, B7 did not work well for me both on CV and Public 

Interestingly se_resnext50_32x4d performed not well on Public LB (0.852) and CV (0.909) but really well on Private (0.702)

As for Decoder I was mostly using Unet++. I have tried [Unet3+](https://arxiv.org/abs/2004.08790) but it showed considerably worse results

As for 3D Nets, I was using [DynUNet](https://monai-dev.readthedocs.io/en/fixes-sphinx/networks.html#dynunet) and adopted model architecture according to [next script](https://github.com/Project-MONAI/tutorials/blob/main/modules/dynunet_pipeline/create_network.py#L19). I have tried to use pretrained Unet from  [MONAI Model Zoo](https://monai.io/model-zoo.html) but it performed badly on all sets 

# Inference and Post Processing 

I was using 512 sliding window with 0.5 overlap, flip TTA, and last checkpoint from 2 folds.
After switching to multi view model, I have also added multi view TTA

The next step was the creation of a kidney mask. I have tried several approaches 
1. Using segmentation net, trained on [this dataset](https://www.kaggle.com/datasets/squidinator/sennet-hoa-kidney-13-dense-full-kidney-masks) + slight post-processing for removing binary holes and small connected regions 
2. Using an algorithmic approach based on intensity thresholding, erosion, and dilation

The first one had a pretty high FP rate but nearly zero FN rate, while the second one had a pretty high FN rate. Both of them performed nearly ideal on kidney_3, so did not really influence the fold 0 scores, but an algorithmic approach cut out kidney regions for kidney_2 and dramatically reduced the fold 1 score. BUT at the same time, the second approach improved Public score (0.874->0.882). I understood that it was 90% overfit to Public LB, but I have decided to take the risk

# Final Model

For final submission, I have selected the following ones:
- Pure 2.5D -> algorithmic post-processing
 - Public: 0.886
 - Private: 0.681
 - Kidney 3 dense score: 0.917   
- 2.5D (weight 3.0) blended with 3D (weight 1.0) 
 - Public: 0.871
 - Private: 0.676
 - Kidney 3 dense score: ~0.918
For both models, I used 0.05 threshold 

# The most popular rubric of this competition: Not Selected Best Submission

Here, I want to point out several of the most exciting approaches for me 
- Pure 2.5D but add Symmetric Focal loss with 0.5 coefficient 
 - Public: 0.867
 - Private: 0.756
 - Kidney 3 dense score: 0.916
- Resize 2d slices to 50 um/voxel for prediction and then resize back 
 - Public: 0.799
 - Private: 0.753
 - Kidney 3 dense score: 0.907
- Resize the whole volume with scipy.zoom to 50 um/voxel for prediction and than resize back 
 - Public: 0.7 resize back 
 - Public: 0.726
 - Private: 0.745
 - Kidney 3 dense score: Have not checked 
- Solo 3D model 
 - Public: 0.849
 - Private: 0.723
 - Kidney 3 dense score: 0.915
For me, it was logical to pick first or second, but as for all other better submissions, it sounds to me like pure random.

# Conclusions

Computing metrics on one data sample leads to severe shakeups 🙂

# Closing words

I hope you have not fallen asleep while reading. Finally, I want to thank the entire Kaggle community, congratulate all participants and winners. Special thanks to Indian University Bloomington, University College London, Yashvardhan Jain (@yashvrdnjain), Claire Walsh (@clairewalsh), the Kaggle Team, and other organizers.
