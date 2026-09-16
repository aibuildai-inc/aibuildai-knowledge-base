# [3rd Place solution] Refine from Sparse to Dense

Competition: blood-vessel-segmentation
Rank: #3
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475074

First and foremost, I would like to extend my gratitude to the organizers and the official Kaggle team for orchestrating such an outstanding competition. I joined the contest at a very late stage. Despite having some experience with segmentation competitions, I must express my appreciation to @hengck23 , @yoyobar , and @junkoda (implementation of metric), as well as the other community participants for their open-source contributions and discussions, which allowed me to quickly get up to speed with this contest.

My approach was strikingly straightforward, relying solely on **2D models** and only utilizing **smp** (segmentation models pytorch) and **timm** (pytorch image models) in the whole training and inference pipeline. 

## Global key points
1.   **Refining labels from sparse to dense.**
2.  **Emulating the magnification factor of the test set.**
3.  **Maintaining an appropriate resolution.**

## 1. From Sparse to Dense
Given that half of the training set has dense labels (kidney 1, kidney 3 dense), and the other half was sparse, utilizing dense labels to refine sparse ones was a crucial step. The overall process entailed:

1. Training UNet(maxViT512) and UNet(EfficientNetv2s) using kidney 1 and kidney 3 dense.
2. Generating supplemental labels for kidney 3 sparse using the trained UNet maxViT512 and UNet EfficientNetv2s models.
3. Resuming the training of UNet maxViT512 and UNet EfficientNetv2s for a few epochs with kidney 1, kidney 3 (dense, sparse plus supplemental labels).
4. Repeating the step2 on kidney 2.
5. Training three UNet models (with EfficientNetv2s, SeResNext101, MaxViT512) and one UNet++ using all real labels from all kidney plus pseudo labels.

Note: As the organizers disclosed the proportion of annotations within kidney 3 and kidney 2, I endeavored to select thresholds based on pixel quantity as close as possible to the official proportion when choosing threshold values for pseudo label.

## 2. Emulating the Magnification of the Private Test Set
A pivotal reason for my decision to participate in this competition was the disclosure of the magnification factors for the training and test sets by the hosts. The training set had a magnification of 50um/voxel, the public test set was the same at 50um/voxel, while the private test set was at 63um/voxel. A larger magnification factor implies a lower resolution. For instance, a 600um object would occupy 12 pixels in both the training and public sets, but only 10 pixels in the validation set. Hence, during training, **I set the scaling center to 0.8**, rather than 1, with a scaling range of 0.55 to 1.05, to simulate the private test set.

```
A.ShiftScaleRotate(shift_limit=0.3,
                    scale_limit=(-0.45, 0.05),  
                    rotate_limit=45,
                    # value=0,
                    border_mode=4,
                    p=0.95),
```

## 3. Maintaining an Appropriate Resolution
In this competition, training and inference along the x-axis, y-axis, and z-axis separately was a very important trick. However, this introduced a significant risk. The entire test set contained 1500 slices, with the public test set accounting for 67% and the private test set for 33%. This means that the private test set comprised only about **500 slices**. Inferring along the z-axis with a higher resolution (e.g., 1024) was feasible. But if inferring along the y-axis or x-axis, it would mean that one of the edges would only be 500 pixels long. At that point, if the model and code were configured for a larger resolution (say 1024), there would be a substantial risk of a huge shake down.

My models primarily operated at a resolution of 512, with one model switching to higher resolution weights for larger resolution slices when the slice have appropriate resolution.

| Model | Backbone | Resolution | public | private|
| --- | --- | --- | --- | --- |
| UNet | MaxViT-Large 512 | 512 | 0.846 | 0.727(submission1) |
| UNet | SeResNext | 512 | 0.819 | 0.753 |
| UNet | Efficiennet_v2_s | 448, 832 | 0.799 | 0.703 | 
| UNet++ | Efficiennet_v2_l | 512 | 0.817 | 0.692 |
| ensemble | - | - | 0.846 | 0.727(submission2) | 


## 4. Train on all data if convergence is Stable
During the early stages of the competition, whether validating on kidney 2 or kidney 3, I observed that if I trained for 20 epochs, after the initial few epochs, the dice coefficient (not surface dice) variation on the validation set was very minimal, with the MaxVit512 large model exhibiting the least fluctuation. Considering that we only had three kidneys, I decided to train on all kidneys directly after completing the pseudo labeling process, given the stability in convergence.

## 5. Minimizing the Impact of Threshold Values
I am grateful for the method provided by @junkoda for calculating metrics. My most stable single model was able to maintain very minor fluctuations in the surface dice score (less than 1) within a threshold range of 0.2. After model fusion, the stable threshold range could be potentially in 0.3~ 0.4. A stable threshold is extremely crucial in segmentation competitions. In this competition, as my final model lacked a validation set, I had to utilize thresholds searched with earlier trained models that included a validation set and apply them to the final version of the model. Fortunately, the models trained on the full dataset appeared to possess threshold values very close to those from the earlier models trained with k1+k2 (sparse), and validate on k3. At the same time, the fluctuation of threshold values across kidney 3 dense, public, and private was very small.

## 6. Heavy augmentation on intensity.
As mentioned by @hengck23 , difference kidneys has large variance on intensity. So I used a heavy intensity augmentation.
```
A.RandomBrightnessContrast(p=1.0),
A.RandomGamma(p=0.8),
```
## 7. Quick ablation
| Model | Backbone | points mentioned above | public | private|
| --- | --- | --- | --- | --- |
| UNet | MaxViT-Large 512 | 3, 5, 6 | 0.818 | 0.586 |
| UNet | MaxViT-Large 512 | 3, 4, 5, 6 | 0.857 | 0.633 |
| UNet | MaxViT-Large 512 | 2, 3, 4, 5, 6 | 0.849 | 0.652 |
| UNet | MaxViT-Large 512 | 1, 2, 3, 4, 5, 6 | 0.846 | 0.727 |

## 8. Final Submission
My final submissions were a single model of MaxViT and a ensemble of the four models. Surprisingly, both submissions scored same at 0.727.  I did not use any form of weighting and MaxViT only constituted a quarter of the ensemble submission, but their scores were totally the same on private LB. Even more astonishing was that the single-model score of SeResNext on private LB turned out to be the highest. Its cv was nothing extraordinary, its convergence was not more stable than MaxViT's, and its public leaderboard score was not high, so I had no reason to choose it.

[1]

Finally, I would like to extend my gratitude once again to the organizers, Kaggle, and all the participants again!

------------------------------------------
**Inference(Submission)** Code is published:
1. [MaxVit512 scored 0.727](https://www.kaggle.com/forcewithme/sennet-final-submission2)
2. [Ensemble submission scored 0.727](https://www.kaggle.com/code/forcewithme/sennet-top3-final-submission?scriptVersionId=162311084)

**Training** Code is published in the [kaggle dataset](https://www.kaggle.com/datasets/forcewithme/sennettop3-training-code/data).
