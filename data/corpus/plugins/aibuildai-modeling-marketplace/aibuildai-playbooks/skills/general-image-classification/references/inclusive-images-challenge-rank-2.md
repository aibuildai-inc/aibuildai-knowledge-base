# 2nd place solution overview

Competition: inclusive-images-challenge
Rank: #2
Source: https://www.kaggle.com/c/inclusive-images-challenge/discussion/72450

I don’t know about other people, but when I saw $1,000 in my inbox I had Christmas Morning moment. Free GCP credits are extremely important for our community. Many thanks to Google &amp; Kaggle!

## Solution
Ensemble of 3 Deep CNNs: ResNet50, InceptionV3, and InceptionResNetV2.

## Concept
We expected test images to come from different regions. So there were no doubts that augmentation and ensembling would be the most important directions to make robust models. But later when detailed EDA showed substantial difference in train/test label distributions it became clear that tuning models using tuning set would also play very important role. 

## Diversity
To get more diversity for ensemble I used following approaches: 

1. Different CNN architectures
2. Different image sizes
3. Train with/without augmentation
4. Different augmentation schedules

## Models
Of course all models were trained from scratch according to the rules. I think it was nice because there was no those usual pretrained model search rush :) Scores below are from Stage 1 LB.

1. ResNet50, 57 epochs, 224x224, adam 0.001, batch 64, increasing augmentation schedule, 4 hours per epoch on 1x P100, 0.311 without tuning, 0.510 with tuning
2. InceptionV3, 25 epochs, 400x400, adam 0.001, batch 32, increasing augmentation schedule, 9 hours per epoch on 1x P100, 0.327 without tuning, 0.545 with tuning
3. InceptionResNetV2, 8 epochs, 598x598, adam 0.001, batch 12, no augmentation, 38 hours per epoch on 1x P100, 0.318 without tuning, 0.550 with tuning

## Final submission
I trained 3 models described above, tuned them, predicted original test images, predicted mirrored test images, and computed weighted average.

## Notes
1. By increasing augmentation schedule I mean the following: train without augmentation until plateau (or until you feel like enough), then apply soft augmentation (wide range of values) during several epochs: e.g. `rotate(np.random.randint(-30, 30))`, then apply hard augmentation (short range or even two possible values) during another several epochs: e.g. `rotate(np.random.choice([-30, 30]))`.
2. About training times. I used batch generator derived from [Sequence](https://keras.io/utils/#sequence) and multiprocessing with 8 processes per GPU. My typical hardware setup on GCP looks like: 1x P100, 8x CPU, 15 GB RAM, SSD or 2x P100, 16x CPU, 30 GB RAM, SSD, etc.

## Side Notes
1. Time after time I had been running 8x V100. It definitely feels like a ride on racing motorbike. I mean not only extreme speed of computation. I also mean adrenaline related to the speed of money flow.
2. Looking at some images I was curious about one thing. How often do you experience something like nostalgia about the country you’ve never been?
3. And by the way have you seen [convolved foxes](https://www.kaggle.com/c/inclusive-images-challenge/discussion/69789)?
