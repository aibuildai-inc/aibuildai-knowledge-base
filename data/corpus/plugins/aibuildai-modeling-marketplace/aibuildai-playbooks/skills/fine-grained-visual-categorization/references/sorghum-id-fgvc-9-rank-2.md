# 2nd Place Solution

Competition: sorghum-id-fgvc-9
Rank: #2
Source: https://www.kaggle.com/c/sorghum-id-fgvc-9/discussion/329414

We really appreciate organizers of this competition. 
Congratulations to all the winners!

# **Dataset**

Only the data released in the competition of this year is used.

We believe that high resolution is beneficial to the Fine-Grained Classification. However, we still test on some lower resolutions because of the limited computation resources. Finally, 960 cropped from 1024 is used in our best solution.

#**Algorithm**

1.	RegNetY-16.0GF is used.
2.	AutoAugment, RandomCrop, RandomHorizontalFlip are used.
3.	Ensemble is performed as usual when different training parameters are used, such as initial learning rate, number of epochs, dropout ratio, etc...
4.	Pseudo Label is also helpful to improve the accuracy iteratively.
5.	TTA is performed during the test.
6.	Dropout is used to avoid overfitting.

#**Results**

| Methods | Public Score | Private Score |
| --- | --- |
| Base（RegNetY-16.0GF, 512x512）| 85.2 | 84.1 |
| 512x512->960x960 | 92.1 | 91.9 |
| Pseudo Label | 95.5 | 95.1 |
| Dropout | 95.7 | 95.3 |
| Ensemble | 96.2 | 95.9 |
