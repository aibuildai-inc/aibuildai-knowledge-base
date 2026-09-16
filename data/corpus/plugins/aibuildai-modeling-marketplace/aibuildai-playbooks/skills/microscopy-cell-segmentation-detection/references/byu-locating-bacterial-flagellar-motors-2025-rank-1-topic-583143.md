# 1st Place - 3D U-Net + Quantile Thresholding

Competition: byu-locating-bacterial-flagellar-motors-2025
Rank: #1
Source: https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583143

Thanks to BYU and Kaggle for hosting this competition. It was nice to have another well-run tomography competition and the hosts were awesome. I can't believe the result!

## TLDR

My solution uses a 3D U-Net trained with heavy augmentations and auxiliary loss functions. During inference, I rank each tomogram based on the max predicted pixel value and use quantile thresholding to determine if a motor is present.

## Cross Validation

For validating models, the competition data is split into 4 folds. Local CV strongly correlates with the LB up to about 0.93. Beyond that, I used the public LB for validation. It was important to use quantile thresholding to get reliable feedback from the LB. More on this in the post-processing section.

## Preprocessing

Tomograms from the competition data and the CryoET Data Portal are used to create a training set. Each tomogram is resized to (128, 704, 704) using `scipy.ndimage.zoom()`, and tomograms without motors are discarded. As others noted, the competition data is quite noisy, so Napari was used to manually add missing motors. I will add the updated data [here](https://www.kaggle.com/datasets/brendanartley/cryoet-flagellar-motors-dataset).

For the labels, I use a Gaussian heat map centered on each motor. Similar to @bloodaxe and @christofhenkel's solution in the CZII competition, the resolution of the heatmap is reduced by 8x. This works especially well for this competition as there is a high tolerance for distance error in the metric. This means that predicting the exact pixel is not as important as predicting motor presence. If you are not convinced, the following plot shows roughly how much error is allowed around each motor when voxel spacing equals 10.

[tomogram_image]

## Model

The model is a 3D U-Net (sort of). The encoder is a pre-trained ResNet200 from Kenoshara’s repository [here](https://github.com/kenshohara/3D-ResNets-PyTorch). For most experiments, I used the ResNet101 variant, but increasing the capacity of the encoder yields better performance. In addition, stochastic dropout is applied for regularization, and gradient checkpointing is used to reduce vRAM usage during training. The decoder uses a single deconvolution block before the segmentation head.

[model_image].jpg?generation=1749095150932706&alt=media)

## Loss

The model is trained using SmoothBCE loss with 3 contributions. The main segmentation head predicts the output logits, a deep supervision head is applied to the second last feature map, and a max pooled loss (kernel size and stride of 4) is applied on the main segmentation head. Moreover, the pooled loss encourages high probabilities around the motor region, while reducing the penalty for small localization errors.

[loss_image].jpg?generation=1751317472394799&alt=media)

## Augmentations

Heavy augmentations enabled training for 400 epochs without overfitting. Although, I could probably have trained longer there was no change in the public LB scores beyond 250 epochs.

- Mixup (100%)
- Rescale/Zoom (100%)
- Rotate90/180/270 (100%)
- Axis Flips (100%)
- Axis Swap (100%)
- Coarse Dropout (50%)
- Color inversion (25%)
- Simple Cutmix (15%)

Loading tomograms from disk is slow, which limits the time for augmentations on the CPU. To address this, all augmentations but rescaling are applied on the GPU. To keep rescaling as fast as possible `scipy.ndimage.zoom(..., order=0)` is used. 

## Inference

Initially, the same preprocessing pipeline was applied during inference. This worked well, but it was 4x faster to match the patch height and width, and only slide over the depth. This allows more time for TTA and a very high overlap (0.875). Both approaches scored about the same, but my final solution uses the latter.

All edge predictions are down-weighted using the `roi_weight_map` parameter. The middle 40% of the logits are weighted as 1.0 and other logits are weighted as 0.001 when aggregating the sliding window.

## Ensembling

The final submission uses an 8-seed ensemble. Sigmoid is applied to each model output and the logits are summed. Inference takes ~10 hrs.

## Postprocessing

Like many others, I found that fixed thresholds were unstable. Instead, I use quantile thresholding to determine motor presence.

To apply this, all tomograms are ranked based on their max predicted pixel value. Then, predictions for the lowest quantile are removed. I tuned the quantile on the public LB and then prayed to the Kaggle gods that the private LB was similar. On the public LB the optimal threshold was 0.565 and on private it was 0.560. 

[LB_image]

## Final Note

Thanks for reading, and thanks to everyone who showed their appreciation for the external dataset. 

External data [here](https://www.kaggle.com/datasets/brendanartley/cryoet-flagellar-motors-dataset)
Github repository [here](https://github.com/brendanartley/BYU-competition)
Metadata [here](https://www.kaggle.com/datasets/brendanartley/solution-ds-byu-1st-place-metadata/data)

Happy Kaggling!
