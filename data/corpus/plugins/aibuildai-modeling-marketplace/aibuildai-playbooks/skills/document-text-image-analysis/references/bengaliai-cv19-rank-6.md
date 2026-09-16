# 6th place solution

Competition: bengaliai-cv19
Rank: #6
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136011

At first, congratulations to the winner @linshokaku and all participants who finished this competition.  
We did not imagine such a big shake ( We estimated that there would not be so much unseen data in private test dataset ).  
  
Anyway thank you Bengali.AI and kaggle for organizing this competition, also to my teammate @nejumi !

---

Below figure shows an overview of our solution.  I think our solution is very simple and nothing special.   
( BTW 1st place solution is very unique and impressive for me ;-)  )
Here I will add naive explanations to some points worth to be noted.  



NOTE: If you want to see a higher resolution image, please refer this [link](https://speakerdeck.com/hoxomaxwell/kaggle-bengali-dot-ai-6-th-place-solution).


1. Intensive augmentations
We used CutMix, MixUp, Cutout, width/height shift, rotate, Erosion, [GridMask](https://www.kaggle.com/haqishen/gridmask) ( credit to @haqishen ), Zoom...  
CutMix and Cutout worked well for both of us, but in my case, height shift and rotation did bad, maybe due to 137 x 236 size (some graphemes become outside image region :-( ).  
<br>
2. Co-occurence
There are limited combinations between R, V and C in train dataset. But as a host said, in test dataset, some unseen combinations will come. So I did a simple trick; build a model which consider both co-occurence and non-co-occurence of R, V, C to be robust with unseen data in SE-ResNet50 based model. This model has dual paths after SE-ResNet block. One is straightforward path from GeM2D to each top fc layers (R, V and C). And the other is through 2 fc layers (please see above figure).
<br>
3. Multi stage learning ( Xentropy, Reduced Focal Loss (credit to @phalanx ) )
In the middle point of this competition, @phalanx advised us about OHEM in this nice [thread](https://www.kaggle.com/c/bengaliai-cv19/discussion/123198#734490).  
Indeed, our cv score (recall) became better with the same technique(Reduced Focal Loss). 
<br>
4. Different image size ensemble
In some CV competitions, different size image ensemble boosted scores. We thought this will be same in this competition. This ensemble pushes our place from 12th to 6th.
<br>
5. Post Processing (big impact for us)
We adjusted our prediction to improve evaluation metric in this competition (Recall). Optimization was performed with all classes (168 + 11 + 7). And we used (1/each class counts) as initial values for optimization. The optimal coefficients are almost close to initial values. We were very nervous to use this post processing, but as a result this pushes us to gold range.
