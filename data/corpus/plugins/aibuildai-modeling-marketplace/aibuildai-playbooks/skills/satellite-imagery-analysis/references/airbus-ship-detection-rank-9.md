# 9th place solution

Competition: airbus-ship-detection
Rank: #9
Source: https://www.kaggle.com/c/airbus-ship-detection/discussion/71595

Congratulations to the winners, especially Victor and Selim of topcoders, whose 2018 DSB winning [solution][1] we adopted for this competition.

## Finding overlap
Due to the overlapping issue, we wrote an algorithm to find all the overlapping pairs of images in train set, then ran the [friend circles][2] algorithm to find all the 57877 groups of images. Images in each group is supposedly from the same acquisition and images from different groups do not overlap.

We then did the train/val split to have 2920 groups (10442 images) in val set while making sure train and val have same distribution of number of ships per image. We did this split before test_v2 was released and used it for the majority of the completion but switched to a balanced 4 fold towards the end which have distribution (in ship number and ship size) closer to test set.

## Binary classification model
An ensemble of fastai resnet34 and resnext50 models trained on our split based on Iafoss’s [kernel][3].

## Labels
We used topcoder’s dsb2018 winning solution code to generate contour (i.e. divider) layer and used 3 channel labels: ship, contour and background. Contour layers are generated in order to help the model separating close ships. An example:
![label][4]

## Model
Best model is Unet densenet169 implemented in Keras by topcoders. We also tried resnet152, resnet101, inception resnet v2

## Loss
Double head loss (i.e. half binary cross entropy, half dice loss) for ship layer. For contour layer, we tried both double head loss and only dice loss since the pixels are too imbalanced.

## Input size
512 and 768

## Thresholds
1. Best classifier threshold is 0.5–0.7 depending on the model. Images below this threshold have empty prediction in submission; images above this threshold go to Unet model which may still end up having empty predictions from Unet. 
2. Unet pixelwise threshold
We noticed that for all our models, best pixelwise threshold on local validation (usually around 0.5) is never the best on public LB. After some simulations and LB probing, we highly suspected discrepancy between train and test label standards: test set masks are more “tight”, i.e. higher threshold on test set (0.6-0.8 depending on model) works out better on public LB. This turned out to  be the case just for the public LB.

## Training techniques
 All the standard ones: data augmentation, cyclic learning rates, TTA

## Postprocessing (mostly using skimage)
-	Algo to separate "weakly connected" masks, with erosion and watershed. We start from an intuition that generally masks are rectangular and when the model predicted ship boundaries are 'weak', they touch along the longer side of rectangles. To separate this case, we first create a thin rectangular structuring element that is aligned with the major axis orientation of the original mask. Applying erosion with this structuring element will help separate the boundary. Since erosion reduces the pixels from the original mask, we try to make this up by applying a watershed algorithm on top of erosion. Example:
![weakly][5]
-	to cut corners (probing the possibility that test set labels may be made as actual ship shape instead of rectangles) 
-	to “rectanglize” unet output masks by finding minimal bounding box and shrink it to same size

## Final models
Best one in local validation and best on public LB. The best on public LB severely overfitted, like quite some teams did. The best one in local validation turned out to be 9th place. 

## Other things tried
-	Ensembling different models by averaging predicted probability matrixes – not working well 
-	Combining different models based on number of ships per image and ship size (some models work better for certain images or ships) – may overfit public LB
-	MaskRcnn – much worse than unet
-	Adding ships predicted in one model to the results of another model – slightly improved but too complicated

## Team and hardwares
We are full time data scientists and coworkers. We don’t do image tasks at our job. This is the first deep learning competition for most of the team members. We teamed up very early. We have 6 Nvidia 1080Ti or equivalent and 2 smaller ones


## Questions for other top teams
- How do you deal with competitions like this where the public LB is too small and delta between local CV and public LB too unstable? Do you always faithfully trust your local CV no matter what? 
- What if the delta is because of difference in distribution (or even labeling standards as discussed [here][6]) between train and test sets? 
- I noticed bestfitting submitted many times, so there's must be some value in public LB?


  [1]: https://github.com/selimsef/dsb2018_topcoders
  [2]: https://leetcode.com/problems/friend-circles/description/
  [3]: https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection/notebook
  [4]: https://i.imgur.com/KanhOpD.png
  [5]: https://i.imgur.com/rYRXeLQ.jpg
  [6]: https://www.kaggle.com/c/airbus-ship-detection/discussion/70221
