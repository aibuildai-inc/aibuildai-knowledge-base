# 41st place solution

Competition: blood-vessel-segmentation
Rank: #41
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/478109

Firstly, we would like to express our gratitude to Kaggle and the organizers for hosting this exceptional competition. Through participating in this contest, we have gained a deeper understanding of the challenges and methodologies involved in medical image recognition.

## Introduction
We submitted separate solutions within our team.
- I submitted an ensemble model of se_resnext101_32x4d and Vision Transformer (mit_b2), which achieved a score of **0.834** on the public leaderboard. The private leaderboard score was **0.586**.
- @ryosukesaito submitted an ensemble model of EfficientNet and SE-ResNeXt which achieved a score of **0.857** on the public leaderboard. The private leaderboard score was **0.519**.
- The high public leaderboard score achieved by @ryosukesaito’s submission might have been a contributing factor to our ability to submit my somewhat ambitious notebook, possibly leading to our winning a silver medal.

## My (@jooott) Solution

### Overview


### Key points

I struggled significantly with stabilizing the training process.
- To address this, I used Accumulate Grad Batches to effectively increase the batch size to 128, which stabilized the training.
- A major factor in the significant improvement in score was the application of stronger data augmentation. The data augmentation strategy was inspired by [the 1st place solution of the Vesuvius Challenge - Ink Detection](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417496).
- I also think that scaling up the training images from 512px to 1024px contributed to the increase in score.





```
train_transform = A.Compose(
            [
                A.RandomScale(
                    scale_limit=(1.0, 1.20),
                    interpolation=cv2.INTER_CUBIC,
                    p=0.1,
                ),
                A.RandomResizedCrop(
                    image_size,
                    image_size,
                    scale=(0.8, 1.0),
                    p=1
                ),
                A.RandomBrightnessContrast(p=0.75),
                A.ShiftScaleRotate(p=0.75),
                A.OneOf([
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                        ], p=0.4),
                A.CoarseDropout(
                    max_holes=1, max_width=int(image_size * 0.1),
                    max_height=int(image_size * 0.1),
                    mask_fill_value=0, p=0.5),
                A.CLAHE(p=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.05),
                ToTensorV2(transpose_mask=True),
            ]
        )
```

## Muku's (@ryosukesaito) solution

.png?generation=1708327848313831&alt=media)

### key points
- In my architecture, Detection/Segmentation of kidney region is performed before predicting blood vessel area.
    - Detection contributed to inference speedup (especially in the yz/zx direction), since it is possible to skip vessel segmentation in frames where kidney is not detected, and to reduce image size by cropping.
    - Segmentation masks were used to reduce FP outside the kidney.
    - For both annotations, I used LangSAM [(luca-medeiros/lang-segment-anything: SAM with text prompt](https://github.com/luca-medeiros/lang-segment-anything)). This allowed me to prepare annotation data with a few manual adjustments.
.png?generation=1708327892920221&alt=media)
    - I use YOLOv8n for Detection and EfficientNet-B0 for Segmentation.

- Various pre/post processing improved LB/PB scores slightly, but steadily.
    - In the yz/zx axis image, blood vessels at the edge may be cut off. Since the inference accuracy was poor in this area, I improved the inference accuracy by pseudo-closing the vessels with mirror-padding before inference.
.png?generation=1708327958129459&alt=media)

    - After binarization of the results, defects may occur in the vascular prediction region as shown below. For this reason, morphological closing and fillPoly processing were added as post-processing steps.
These contributed to a slight score improvement in CV/LB/PB.
    .png?generation=1708328014528398&alt=media)

- In my experiments, ideas that contribute to generalization ability (strong augmentation, pseudo labeling, etc…) could not adopted as final submits, because they resulted in a decrease in CV/LB…
However, I regret that I should not have been too aware of the unstable CV/LB, as the sample was not large enough for this competition.
