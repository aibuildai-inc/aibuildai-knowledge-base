# 4th Place Solution: MILIMED

Competition: hpa-single-cell-image-classification
Rank: #4
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/239071

We are a very diverse team of computer scientists and medical doctor/students. It was our great pleasure to participate in this demanding challenge. Hope some of you find this solution useful and/or interesting.

# Solution overview
1. Segmentation -> HPA-Cell-Segmentation
2. Dataset -> 512x512 cell images (20% removed)
3. Parallelization -> speed-up -> 3h left for inference
4. Manual Labeling -> smaller classes & validation (soft labels)
5. Pseudo-Labeling -> negative labeling (& positive for mitotic spindle)
6. EfficientNetB0 Ensemble + semi-balanced data sampling
7. Fine-tuning -> on manually labeled & non-labeled validation data
8. Cell/Image Weighting -> final confidence = 0.7 \* cell\_confidence + 0.3 \* image\_confidence

# 1. HPA-Cell-Segmentation

The test set was based on this segmentator so it made no sense to spend a lot of time creating a custom segmentator which could make the IoU worse. The authors of the contest said that only 10% modifications were made on the outputs of the segmentator.

# 2. Dataset

Our dataset was created from the Train & PublicHPA 16-bit images. Seems most teams used 8-bit images in the end.

Each image in the final dataset is a 512x512 image of a cell based on the cell masks from the segmentator. Padding (to square) was used to retain original height/width ratio. No surrounding pixels were used (non-cell-mask pixels). My feeling is that it might be better to use a bit larger surrounding, but did not have time to test this (this might be good for some classes such as plasma membrane).

We decided to go with large images (512x512) since some labels/organelles required higher resolution and their size varied a lot based on the cell size & re-scaling. E.g. sometimes the nucleus was very small and sometimes it as big as the whole image. We even tried to train nuclear organelles on images based on nuclei masks, but since the deadline was too close, we decided not to invest more time on this approach.

I am eager to find out if diving the problem into nuclear and cytosolic organelles classification would yield better results. I think it would be easier to classify organelles inside the nucleus since it would be approximately the same size for each cell image this way.

We used a simple heuristic to determine how much of the nuclei was outside of the image and decreased its final predicted confidence accordingly. All images with nuclei that were not present almost completely in the cropped image were removed from the train set.

Similarly we tried to determine false positive segmentations by finding outliers based on the red channel and a product of the blue and yellow channel. Outliers at inference time got their confidence decreased dramatically. Outliers in the train dataset were removed completely. I assume the accuracy of this heuristic was around 50%. Since false positives were a big score crusher, this seemed acceptable.

We lost around 20% of the images from the trainset.

# 3. Parallelization

HPA-Cell-Segmentation took quite some time so we decided to parallelize most things. Even with only two cores we got a boost in the submission time.

The first boost was by running the `label_cell` function in parallell. The second boost was in running all the previously mentioned heuristics and image cropping in parallel as well.

This left us with more than 3 hours for inference.

# 4. Manual labeling

We manually labeled smaller classes or classes with smaller % of occurrence in the initial images (e.g. mitotic spindle, aggresome, intermediate & actin filaments ...). We made a simple GUI and relabeled only one label at a time for an image.

Mostly we would give a score from 1 to 5 on how confident we were that the given cell image contained the image-level label. These scores we transferred to soft labels. Each mapping was different (e.g. 1:0.0, 2:0.2, 3:0.7, 4:0.9, 5:1.0).

In the end, we tried to create a validation set in the same way with high quality labeling. We managed to do get a few thousand examples for most classes.

# 5. Pseudo-labeling
Inspired by the Meta Pseudo Labels paper, we wanted to get rid of some false positives and help our models avoid overfitting. A cut off of 0.3 seemed to remove approx. 15% of image with high accuracy. Here we used an underfitt ResNet18.

Later we used a better model to find more examples of mitotic spindles in a similar way, but withing the images that did not have mitotic spindle assigned. I think we found around 100 extra mitotic spindles, compared to around 250 that we found in the labeled images.

In the end, we did not do this for other classes. I think we found a few aggresomes and quickly decided to skip positive labeling.

# 6. EfficientNetB0
This network is just awesome :) I am a big fan of solving problems with simple models, so I was quite happy when EfficientNetB0 seemed to be good enough for this challenge. We tried using B4, but it was slower to train and the results did not impress enough to continue playing with it. There was a solution that ensembled some B4-s, but no boost on the private LB.

We had an 3-part ensemble with weights [0.2, 0.4, 0.4]. All EfficientNetB0s, but trained with different augmentation and loss function combinations.

1. Single B0 - 0.2 ensemble weight
Augmentation: Flipping & Rotation
Loss: FocalLoss
Did not have time to test if this network actually helped much.

2. 2 Checkpoint Ensemble B0s - 0.4 final ensemble weight
Augmentation: RandomResizer (40% chance), Flipping & Rotation
Loss: FocalLoss
*RandomResizer -> Resize to (RSIZE, RSIZE) + Resize back to (512, 512), where RSIZE is a random number between 256 and 384

3. 4 Checkpoint Ensemble B0s - 0.4 final ensemble weight
Augmentation: RandomResizer (30% chance), RandomPad (30% chance), Flipping, Rotation, & Resize(512,512)
Loss: BCELoss
*RandomPad -> pad each side (independently) with a random length between 0 and 200

Since it was hard to estimate the "best" model on the local validation set, we used the idea of checkpoint ensembling to try to avoid overfitting & boost our score.

We oversampled classes with less examples or less true positives and tried to avoid overusing images with more assigned labels. 

# 7. Fine-tuning

We fine-tuned all networks for one "epoch" on the validation set. Unlabeled images were used once, while images that we labeled (soft labels) were used multiple times in the "epoch". The "epoch" was around 200-250k images. We used the same augmentations & loss function for each B0 as it was used during training of that network.

# 8. Cell/Image weighting

To extreme outliers we weighted the final confidences with the mean of confidences of all other valid cells (excluding border and outliers). The final confidence was 0.7 \* cell\_confidence + 0.3 \* image\_confidence.

Seems that even extreme values such as 0.6/0.4 tend to work well here. We did not test if smaller weighting gave better score.

# Conclusion

512x512 images, re-labeling, simple network (B0), fine-tuning & final confidence weighting seem to work well enough for this problem.

# Thank you
@christofhenkel & @cwinsnes -> [https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/229284#1258324](url)
We were not aware of this at that time. This valuable responses made a huge impact on our approach/results.

@h053473666 -> [https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/230940](url)
Thank you for the motivation for the final weighting.

@emmalumpan, @lnhtrang & @cwinsnes I hope there will be more opportunities to participate in your journey in the future (Kaggle or non-Kaggle related). Thank you for the ride! :)
