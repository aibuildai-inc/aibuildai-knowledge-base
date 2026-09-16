# 7-th place solution

Competition: severstal-steel-defect-detection
Rank: #7
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114215

Our team member [xuyuan](https://www.kaggle.com/xuyuan) and I had already participated in a segmentation challenge (TGS salt) and we thought this time will be a little easier, so joined this competation really late 4, 5 weeks ago. But these few weeks was not enough to try out many things. We haven't even time to train properly a classifier, so we stick to a pure segmentation ensemble.

Our final ensemble contained 3 efficientnet unet models:
* 2 folds of efficientnet-b7 unet. trained on 256x384 with mixup and label smoothing. The best public score was 0.91288. We don't know the private score because only the csv file was submitted. In the final ensemble, these b7 models was fine tuned on the full resolution.
* 1 fold of efficientnet-b4 unet. trained on 256x384 and fine tuned on the full resolution.
* 2 fold of efficientnet-b0 unet. trained on 256x384 and fine tuned on the full resolution. The best one got public 0.91284 and private 0.89719 score.

For the ensemble we have tried many thresholds. For final submission we have used [0.8, 0.99, 0.75, 0.75] which produced 87, 0, 602, 111 class images. Another submission which used [0.8, 0.99, 0.78, 0.75] produced  87, 0, 577, 111 class images. This would have a private score 0.90819 and public 0.92016 which would be enough for the second place. So predicting the class3 less seems to be a trick.

The min area was 100 and pixel threshold 0.3.

What worked for us so far:
* AdamW with the Noam scheduler
* gitlab pipeline to manage training experiments
* mixup and label smoothing
* fine tuning on the full resolution
* random sampler
* cross entropy loss

What didn't work:
* SGD
* SWA worked really good for the salt competition, but for this competition it didn't work at all
* pseudo labeling (trained on last 2, 3 days)
* training a classifier
* balanced sampler
