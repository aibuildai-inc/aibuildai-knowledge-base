# Finally GM & 7th place solution

Competition: understanding_cloud_organization
Rank: #7
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/117974

congratulations for all kagglers.
# Small talk
After the failure of steel, I had no time to feel sad, so I immediately went to this competition, made efforts for my GM. One of my former teammates [Xuan Cao](https://www.kaggle.com/naivelamb) choose solo. Of course, he made the right choice because he win a  solo gold and became GM. Congratulations to him. 
And then I choose my friends who had been suffered from steel like me, [Zheng Li](https://www.kaggle.com/mdlszhengli), [yelan](https://www.kaggle.com/lanjunyelan), [Jhui He](https://www.kaggle.com/hesene) and [Strideradu](https://www.kaggle.com/strideradu)  to team up.

# Solution
&gt; Our solution is very simple, just ensemble.

## Segmentation v1：
Model: efficientnet e5/7-FPN，se101-FPN, se101-linket
Loss: dice loss

## Segmentation v2:
Model: efficientnet e5-FPN
Loss: SymmetricLovaszLoss+dice loss

## Classification:
&gt; We have tried some pure classifiers, but the dice improvement of oof of our segmentation is limited, so we turn to multi-task learning, a segmentation model with fc head.

Model: efficientnet e5-fpn, se50-unet, se50-fpn
Loss: 0.1 * bce (classification) +(bce + lovasz + dice)(segmentation)

## Ensemble:
### v1: 
Averaged probability from classification model for removing fp, and Segmentation v1 for tp, then we can got the around 0.670 oof cv, and the threshold for classfication is [0.65, 0.65, 0.65 ,0.65], then remove the small size mask(the threshold is [21000, 21000, 21000, 10000]), finally got the 0.6783 lb.
### v2:.
we  averaged probability from classification and the max pixel probability from Segmentation v2 for removing fp, but the lb was bad, and the threshold is low(0.55, the low threshold is not good in steel), so we abandoned this.

## Post processing:
1. From training set, each image has at least one label. Then, I extracted 4 channels with empty samples From my 6783 sub above, and extracted the maximum prediction probability of the classifier on this sample, and then restore the pixel mask if the prediction probability for a category &gt; 0.55
2. I did a mask union of the samples that both Segmentation v2 and 6783sub predicted as postive.
Combined with the above post-treatment, we can get 0.6800 lb.

# Conclusion
1. Unfortunately, we did not select the best submission(0.67254), which was from ensemble v2.  but the public lb was 0.67360, so we did not select this submission, and of course we had a lot of submissions in the top3. Fortunately, Our submission which we choose, still allows us to go into the gold zone.
2. Thanks to my teammates for their efforts and I congratulate myself on becoming GM.
