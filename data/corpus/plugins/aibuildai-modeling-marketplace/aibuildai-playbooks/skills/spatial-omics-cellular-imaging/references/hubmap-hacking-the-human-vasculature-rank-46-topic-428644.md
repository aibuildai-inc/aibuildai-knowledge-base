# Private 47th solution

Competition: hubmap-hacking-the-human-vasculature
Rank: #46
Source: https://www.kaggle.com/c/hubmap-hacking-the-human-vasculature/discussion/428644

Our solution is an ensemble of yolov8m-seg models with different inputs: 512, 640, 1024. We pretrained our models using "unknown" and "glomerulus" as blood vessels since they have similar structures with the target. Then, we retrained our models with the correct blood vessels without any "unknown" or "glomerulus" labels.

For test time augmentations (TTAs), we used rotation (90), horizontal flip, vertical flip, and diagonal flip. Additionally, we lowered the pixel threshold to 0.45, allowing the model to decide where and how to dilate. However, it didn't perform as well alone as dilation on the public leaderboard, so we kept dilation.

To ensemble masks, we used the code from https://www.kaggle.com/code/mistag/sartorius-tta-with-weighted-segments-fusion.

We also utilized DBSCAN to crop images where major predictions are located. This technique was used in the [image_matching comp 2022](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/329131). The aim here is to fine-tune the predicted masks with their confidence( random cropping should also work).

Additionally, we performed fine-tuning of the predicted masks using SAM.

What did not work:
- TTAs of scaling didn't work at all.
- Normalization using staintools didn't work in the public leaderboard (resulted in a drop of about 0.008), while it worked in the private leaderboard (resulted in a gain of +0.01).
Our best private score was 0.561, an ensemble of yolox(512) + yolom(1024) with a pixel threshold of 0.45, no_dilation.


