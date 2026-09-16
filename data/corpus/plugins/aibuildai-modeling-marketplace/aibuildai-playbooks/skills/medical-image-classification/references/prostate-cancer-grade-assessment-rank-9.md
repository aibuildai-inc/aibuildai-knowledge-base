# 9th Place Solution [ResNeSt might be the key]

Competition: prostate-cancer-grade-assessment
Rank: #9
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169141

First of all Thank you very much to organizers.

My solution is simple.
I ended up with the following models in my final ensemble submission.

Label Smoothing + Ordinal regression + efficientnet_b0 x 2
Label Smoothing + Ordinal regression + GeM + ResNeSt50 x 2

I was not confident with efficientnet_b0 since there was difference of about .02-.03 between the CV (.90XX) and Public LB (.87XX). It seemed to me like it was overfitting the LB despite using Label Smoothing and less stable.
Whereas, ResNeSt50(Split-Attention Network) gave me stable difference between Public LB (.85XX) and CV (.86XX) along with Label Smoothing.

We all knew that qwk was not a stable metric for given amount of Public Test Images. We also knew that training data seemed to have ample amount of noise.
So, My intuition was to select one less scoring more stable model and one best Public LB scoring model for ensembling.  
I manually weighted the models during ensembling. I observed that giving more weight to efficientnet_b0 (Best Public LB scoring Model) leads to decrease in Public LB, therefore I selected the ensemble having equal weights (simple average).

After the releasing of Private LB the same Resnest50 gave a private LB score of .90XX.

I was pretty sure to atleast land in bronze level zone, but shake up seemed to be very rough eventually landing me in gold zone. (completely unexpected)

For training, I used both the techniques for tiling the images (List of tiles and Single large Image of Tiles). I used 36x256x256 (level - 1) tiles along with simple augmentations like hflip,flip,transpose,rotation..

**What More Could Have been Done ??**

Since I was using kaggle and Colab for training, I was constrained to 30 hr weekly and 12hr limit respectively. It is clear that I don't had enough gpu compute to do more experiments.

But the following seemed to be much effective approach :- 

I also tried training seresnext50 + AdaptiveConcatPool + Classification (pretty sure that Ordinal Regression would have scored better) on Single Large Image of tiles dataset with noisy labels removed (Images Having Pen marks ) along with **Progressive Resizing** (ie) training in following sequence:-

16x128x128 (Level-2) --&gt; 16x160x160 (Level-2) --&gt; 20x224x224 (Level-1) and so on....

Make sure to load weights from previous stage while training the next stage.
 
Due to the limited compute power I wasn't able to train seresnext50 on larger image size, my final input image size was 20x224x224 which gave me CV of .81XX and Public LB of .82XX (pretty stable).
When the Private LB came, the resnext50 model trained in very same way mentioned above gave private LB of .89XX.
Since I wasn't able to train this on larger images I wasn't able to use it in final ensemble.

I am pretty sure it would have given a decent boost to my final score.
