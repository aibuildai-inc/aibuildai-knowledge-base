# 25th place solution [Kaggle_gaggle]

Competition: prostate-cancer-grade-assessment
Rank: #25
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/171065

First of all Thank you very much to organizers and thanks to @kyunghoonhur for collaborating with me!

We are happy to get unexpected medal(Our PB rank is 124th). Individually, this is my first medal on kaggle and this medal made me even more into kaggle.


Based on @haqishen  train &amp; inference notebook, we will mention some of things we tried and a comment about how those works affect on our final result.

## Tile Size Selection 
 
- 256x256x36 tiles : best results
- 128x128x16 tiles : lower result than 256x256x36 tiles. It seemed necessary to raise the image resolution.
- 256x256x16 tiles : better result than 128x128x16 tiles, but not satisfactory.
- 256x256x36 tiles with little white as possible : Since 256x256x36 tiles have lots of white spaces, we tried to remove white spaces based on @rftexas’s [notebook](https://www.kaggle.com/rftexas/better-image-tiles-removing-white-spaces). It achieves lower train loss than simple 256x256x36 tiles but quadratic weighted kappa score did not improved.

## Augmentation

Several different augmentation were tested (Transpose, VerticalFlip, HorizontalFlip, RandomRotate, Blur, etc), but not much performance improvement was seen.
Just taking basic augmenation configuration based on @haqishen ’s notebook.
Albumentation library
&gt; Transpose(p=0.5)
VerticalFlip(p=0.5)
HorizontalFlip(p=0.5)
All the augmentation were made at 2 levels: tile level + after the tile concatenated

## Model

Similar to other competition (Deep learning for image classification), the most popular model architecture (Resnet, efficientnet) we tried.
Among many several Resnet model structure,  SE_Resnext50 was shown the highest score (except more than 50 model because our GPU limitation).
Efficientnet showed stable and high score at CV.
We couldn't get high level of efficientnet model due to our GPU unfortunately , but some discussion let us know that deep and heavy size model will lead to overfit (Effnet b6)
So we focus on Efficientnet B0 and B1, between them not much difference shown.

## Optimizer &amp; schedular

Adam optimzer 
Adam + GradualWarmupScheduler + CosineAnnealingLR

## Inference

a) TTA(Test Time Augmentation)

Based on tile generation method from Quishen Ha kernel, slight augmentation was added when conducting tile extraction
That code is at mode=0 or mode=1 option of PANDA dataset generation class.
Difference between mode =0 and mode1 is the sequence of tile into the concatenated input (36 x tile).
So, when inferencing model, mode1 tile and mode 2 tile were considered as augmented data for test time augmentation(TTA).
Additionally, we added transform augmentation in the same way  of train (2 levels, tile + concatenated input).
From several experiments, TTA showed quite positive effects on our public score when increasing the number of augmentation data.
However, considering this competition is code competition which limits the submission time below 9 hours,  we made intermediate number of TTA not as much like more than 100 TTA for preventing over of regular submission time.

&gt; 16TTA(mode=0) + 16TTA(mode=1)
Transpose(p=0.5)
VerticalFlip(p=0.5)
HorizontalFlip(p=0.5)

b) Model Ensemble

The hardest part in this competition was how consider overfit on our training data and how predict shake up from private data.
We carefully watched our CV score and LB score and continuously compere them.
At last, from the comparison CV and LB for each fold, we got the fold which had the most similar result between CV and LB score.

Ensemble result [Efficient net b0(fold0) and Efficient net b1 (fold0 and fold1)] showed the best score at public score and final(private) score both.
