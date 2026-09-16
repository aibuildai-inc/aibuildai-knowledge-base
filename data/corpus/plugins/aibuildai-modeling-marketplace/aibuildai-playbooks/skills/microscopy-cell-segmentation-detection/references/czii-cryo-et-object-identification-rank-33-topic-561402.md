# 33 place solution

Competition: czii-cryo-et-object-identification
Rank: #33
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561402

I use Data Augmentation

`random_transforms = Compose([
    RandCropByLabelClassesd(
        keys=["image", "label"],
        label_key="label",
        spatial_size=[128, 128, 128],
        num_classes=7,
        num_samples=my_num_samples
    ),
    RandRotate90d(
        keys=["image", "label"],
        prob=0.3,
        spatial_axes=[0, 1]
    ),
    RandRotate90d(
        keys=["image", "label"],
        prob=0.2,
        spatial_axes=[1, 2]
    ),
    RandFlipd(
        keys=["image", "label"],
        prob=0.3,
        spatial_axis=0
    ),
    RandFlipd(
        keys=["image", "label"],
        prob=0.3,
        spatial_axis=1
    ),
    # Optionally, you can also flip along the third axis:
    # RandFlipd(
    #     keys=["image", "label"],
    #     prob=0.3,
    #     spatial_axis=2
    # ),
    RandAffined(
        keys=["image", "label"],
        prob=0.5,
        rotate_range=(0.17, 0.17, 0.17),
        scale_range=(0.05, 0.05, 0.05),
        mode=("bilinear", "nearest"),
        padding_mode="zeros"
    ),
    RandScaleIntensityd(
        keys="image",
        prob=0.2,
        factors=0.1
    ),
    RandShiftIntensityd(
        keys="image",
        prob=0.2,
        offsets=0.1
    ),
    RandAdjustContrastd(
        keys="image",
        prob=0.2,
        gamma=(0.9, 1.1)
    ),
    RandHistogramShiftd(
        keys="image",
        prob=0.2,
        num_control_points=10
    )
])
`

**Model Architecture**
My model is based on the MONAI UNet and uses the following configurations:

Channels: (64, 128, 256, 256)
Strides Pattern: (2, 2, 1)
Number of Residual Units: 1
Due to limited GPU resources, I utilized an L4 GPU.

**Performance Results**
Pure 3D UNet: Achieved a public score of 0.722 and a private score of 0.726.
Ensemble with YOLO: Reached a public score of 0.757 and a private score of 0.755.

**Additional Enhancements**
I also incorporated a filtering mechanism that ignores any cluster where the standard deviation of each coordinate exceeds 40% of the particle radius.

For further details, please refer to the notebook.
https://www.kaggle.com/code/junhanzangai/czii-cryo-s
https://www.kaggle.com/code/junhanzangai/submission-test\

And i attach my code, too.

Thank you for giving me this opportunity.
