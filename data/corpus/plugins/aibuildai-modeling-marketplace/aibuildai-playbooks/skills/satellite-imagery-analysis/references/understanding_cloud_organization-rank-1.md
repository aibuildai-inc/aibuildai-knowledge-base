# 1st placed solution with code

Competition: understanding_cloud_organization
Rank: #1
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118080

## UPDATE: code available on github
https://github.com/pudae/kaggle-understanding-clouds

---

Congrats to all the winners and survivors of the shake-up.
Thanks to Kaggle and the hosting team for the interesting competition.

Except for some tricks, improvements almost have been made by using ensemble. So, in this post, I will briefly describe the track of scores in the last week. The details will be shared as codes.

### Common Settings
**Types of networks**
- Model A: UNet with classification head
- Model B: FPN or UNet, no classification  head

**Backbones**
- resnet34, efficientnet-b1, resnext101_32x8d_wsl, resnext101_32x16d_wsl

**DataSet**
- split: train vs val = 9 vs 2
- Model A: All labels
- Model B: non-empty labels

**Loss**
- classification part: BCE
- segmentation part: BCE * 0.75 + DICE * 0.25

**Optimizer**
- AdamW, weight decay 0.01
- encoder learning rate 0.000025
- decoder learning rate 0.00025
- OneCycle scheduler, shallow models 30 epochs, deep models 15 epochs

**Augmentation**
- Common: hflip, vflip, shift/scale/rotate, grid distortion, channel shuffle, invert, to gray
- Model A: random crop, size 384
- Model B: full-size, size 384, 544, 576, 768

### The track of scores
**train single model**
At first, I’d tried to train a good single network. I’d struggled to improve and stabilize the LB scores for 2 weeks, but I’d failed. 
- TTA3: CV 0.6517 / Public LB 0..66951 / Private LB 0.65828

**add segmentation models**
I thought the reason for the unstable LB score was because of poor segmentation performance. If we can have a more powerful segmentation model, the effect of poor classification performance can be reduced.

So, I began trying to train good segmentation only model. Because I could filter out negative predictions using the classification model, only positive labels were needed to train.

From this time, CV and LB were correlated well.
I trained several segmentation models with different backbone, image size, etc.
- TTA4, 1 seg with cls + 1 seg: CV 0.6560, Public LB 0.67395, Private LB 0.66495
- TTA4, 1 seg with cls + 3 seg: CV 0.6582, Public LB 0.67482, Private LB 0.66501
- TTA4, 1 seg with cls + 4 seg: CV 0.6587, Public LB 0.67551, Private LB 0.66604
- TTA4, 1 seg with cls + 7 seg: CV 0.6594, Public LB 0.67596, Private LB 0.66663

**add more models with classification head**
Now, the segmentation part became enough good. so, I added two more models with classification head.
- TTA4, 3 seg with cls + 7 seg: CV 0.6625, Public LB 0.67678, Private LB 0.66746

**use segmentation models as a classifier**
To take advantage of the performance of the segmentation models, I used a mean of top K pixel probabilities as a classification probability. 
```
cls_probabilities = np.sort(mask_probabilities.reshape(4, -1), axis=1)
cls_probabilities = np.mean(cls_probabilities[:,-17500:], axis=1)
```

- TTA4, 3 seg with cls + 7 seg: 0.6629, 0.67822, 0.67046
- TTA4, 3 seg with cls + 8 seg: 0.6635, 0.67906, 0.67117

**use max probability as a positive prediction**
All images in the train set have at least one type of cloud, so I treated the label of max probability in each image as a positive prediction. 
```
cls_probabilities[np.argmax(cls_probabilities)] = 1
```
- TTA4, 3 seg with cls + 8 seg: CV 0.6640, Public LB 0.68031, Private LB 0.67170

**use exponential moving average**
Finally, I changed the averaging weights method to the exponential moving average. Before that, the average of the last 5 weights was used.
- TTA4, 3 seg with cls + 8 seg: CV 0.6636, Public LB 0.68130, Private LB 0.67126
- TTA4, 3 seg with cls + 9 seg: CV 0.6637, Public LB 0.68185, Private LB 0.67175 (**Final Submission**)
