# 30th place solution with code

Competition: severstal-steel-defect-detection
Rank: #30
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114225

Wow, I've not expected such a big shake!
This 2~3 weeks I almost gave up this competition because my public LB score was so poor, but it seems to be better not to throw away and I'm so lucky!

---
My solution was not so special, but I want to share it for the case it's useful for beginner or someone who wants to start segmentation competition.

- apply 5 class classification including background class, and then 4 class segmenteation.
- For classification task, I trained resnet50, efficientnet-b3 and se-resnext50.
- For segmentation task, Unet with resnet18, PSPNet with resnet18 and FPN with resnet50.

What worked:
- random crop by 256x800 or full 256x1600
- 5 class classification (including background)
- 4 class segmentation
- BCE + Dice loss

What didn't worked:
- random crop by 256*400
- pseudo labeling
- softmax and cross entropy loss
- bigger model(encoder) was just waste of inference time
- mixup
- cutmix
- thresholds higher then 0.5

---

My code for this competition specific part was mainly base on [this great starter kernel](https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88) by @rishabhiitbhu.
And I also borrowed many many idea from @hengck23 's discussion.

Thank you for all competitors and congrats for all finished in gold zone!
  
 --- 
Here is my code:)
https://github.com/bamps53/kaggle-severstal.git
