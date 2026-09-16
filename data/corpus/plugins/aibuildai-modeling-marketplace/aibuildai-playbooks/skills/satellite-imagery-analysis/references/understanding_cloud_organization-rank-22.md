# 22th Place - Lessons learned from a beginner

Competition: understanding_cloud_organization
Rank: #22
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118184

First congrats to all the winners.
I would like to thank Kaggle for hosting this competition which was the first one I could dedicate myself and won my first medal.

**What didn't work for me**
- Lovasz loss
- Deeper encoders (efficientb7,senet)
- Pseudo labeling

**Our solution**
Our solution is basically emsemble of segmentation models with post processing to remove masks

Models (6 folds each):
- ResNet34 - Unet*
- EfficientNetB2 - Unet*
- EfficientNetB2 - FPN
- EfficientNetB2 - LinkNet
- EfficientNetB5 - Unet

Loss: BCE + Dice
* Those models was trained with different image size (320x480, 384x576, 512x512, 704x1056)

Post Processing:
triplet threshold searching for binarization, remove small masks and binarization again for the remaining masks after the first two steps. All this was done with the validation data from all 6 folds.

CV: 0.6651
Public: 0.67556
Private: 0.66498

**The Good Lesson**
I didn't know much about image segmentation, so this competition was a great learning.

- Read all comments and try to get the tips.

- Build a good validation set
tuning post processing parameters was only possible without overfitting because of that


**The Bad Lesson**
- Trust in your CV

I had a better model that scored:

CV: 0.6681
Public: 0.66759
Private: 0.66824

Why didn't I choose it? because of the second lesson ...

-  Trust in you
My best model was something different. I trained one model for each mask type, predicted one by one and put it in original format (4 masks stacked) before applying post processing. 

This allowed me to compare with the same out of fold data I had so far. 
A simple blend of ResNet34-Unet + EfficientNetB2-Unet got 0.668 on CV.

But I read that some kagglers didn't get good results with this method, I was afraid of having a leak in my validation and public LB was worse. So I gave up on this idea...


**Acknowledgment**
I would like to thank my team and all those who shared in some way.

Sharing is a very good thing, but I think it should be done at the right time. As I said I am a beginner, but also someone who worked hard on this competition reading past competition solutions. So I think everyone can do the same.
