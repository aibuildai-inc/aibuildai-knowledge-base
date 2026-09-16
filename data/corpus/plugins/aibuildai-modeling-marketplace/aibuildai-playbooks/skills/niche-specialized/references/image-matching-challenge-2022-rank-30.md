# 30th Place 827 Private LB with image size 840

Competition: image-matching-challenge-2022
Rank: #30
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328844

First of all i would like to thank the organizers for this wonderful competition.
Great thanks to my teammate and friend @jaafarmahmoud1 for his hard work we really enjoyed working together again here in kaggle.
## What I didn't like on this competition
- The inconsistency between training and testing data. so if you work on validation you will probably end out off the medal zone. I thought that the private testing set will be hard to test the models robustness and stability but it seems really similar to the public data (in terms of the results).
##   Our solution:
We have developed two solutions one is to maximize the validation and another one to maximize the LB. 
### Best validation submission:
After testing super point + super glue, DKM , Loftr, loftr QuadTree and match former  on the validation data. We noticed that some of them impact (increase) easy scenes like "notre_dame_front_facade" but drop the MAA for other fields (Hard - medium) like "colosseum_exterior" which indicates a huge difference in results (shakes) if the data is more generalize.
The difference for the same scene could be up to 0.08.
To overcome this problem we have validated the combination of the models and took the best results for each scene to label the data. We trained a classification model to tell when to use DKM + Matchformer light and when to ignore as well when to use super point + loftr and when to ignore. and the main part (models) which will be applied to all images is an ensemble of match former and loftr QT. 
#### Tricks that will work for everyone:
- Using loftr gives an ability to pass a mask with the input. This mask originally is for padding so if you padded the image from 840 x ? to 840x840 and pass a mask that indicates the padding area to ignore.  the results will become even better up to +0.01
- From the mask idea what will happen if you pass the mask as a depth or segmentation mask instead of just padding mask. That will change the matches and points giving you another ensemble to add and enhance your results up to +0.01.
- input reversed instead of passing the images as inp1, inp2 pass them as inp2 then inp1 you can also flip them to get better results but it is a really good approach to get over the grid problem with loftr models up to 0.015. it will also works for super glue so you can apply super point once with two super glues + 0.006.

This model scored 0.813 on public and 0.816 on private lb.
### Best submission
 Our best submission on lb uses the same models without loftr QT. and the classification model used only to determine if to use DKM or not.  The main model were SP + 2 SuperGlue, loftr, match former reversed and match former light.

## Tried but didn't work:
- Anything not MAGSAC++ was worse including (USAC_ACCURATE, DEGENSAC, VSAC ..)
- MEMORIZE from two images look for a third image that has a higher cov with both and calculate the F matrix
$$X_1^t F_{12} X_2 = X_1^t F_{13} X_3 X_3^t F_{32} X_2 -> F_{12} = F_{13} X_3 X_3^t F_{32} $$
- OANet - Disc - HardNet and (any other public model)
- Multi step MAGSAC
- TTA other than flipping was :( 
- Cropping the area of interest (has the major number of matches) or multiple areas


I hope that my contribution to this competition was helpful thanks to all of you and congrats to all medal winners
