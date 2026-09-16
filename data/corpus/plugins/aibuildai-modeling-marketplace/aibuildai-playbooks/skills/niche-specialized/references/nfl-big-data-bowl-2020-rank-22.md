# Public 14th Solution - Transformer

Competition: nfl-big-data-bowl-2020
Rank: #22
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119381

Thanks to the organizers for such an interesting competition! This is the first time for me to use NN as my main model, and I've learned a lot from this competition ;)

After a quick glance at training data, I thought this competition is quite similar to CHAMPS Predicting Molecular Properties competition - so I decided to use transformer in the early stage of the competiton.

### Model
.png?generation=1574937511409845&amp;alt=media)

My architecture is 3 pointwise convolution layers (like PointNet) plus 4 transformer encoders followed by few dense layers. I tried multi-headed attention, but it didn't work for me. Implementation of transformer is based on PyTorch 1.3, but slightly modified (change post-LN architecture to pre-LN. see this paper in detail [here](https://openreview.net/pdf?id=B1x8anVFPr)).

### Features
I use 17 features in total. Except for YardLine and Distance, 15 features are calculated for each player (so 15 x 22 feature vector are fed into transformer encoder). All features are quite simple, but the most important one is an angle from Rusher's moving direction. 
 


This polar coordinates based on rusher's dir gave me a big boost (~0.0003). I also calculate this angle in 0.001s after the given time and calculate the difference between θ(t) and θ(t+0.001).  

List of all features used:

- YardLine
- Distance
- X, Y, S, A, Dis, PlayerWeight
- IsBallCarrier
- IsOffence
- sin(Dir), cos(Dir)
- dX, dY, atan(dY, dX) from Rusher
- angle, angle variation from Rusher's direction  (described above)

That's all :) CV and Adversarial validation is used to remove irrelevant or suspicious features.

### Validation
In addition to standard GroupKFold, I also tried time-split validation (discussed in https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/113861#latest-658998). Both worked well for me but its validation score differs a lot. So I choose one from each validation strategy for the final submission.

### Other techniques (maybe) worked
- AdamW optimizer (slightly better than Adam)
- Gaussian noise and Y-shift augmentation on player vector
- TTA (20x augmenting and adding gaussian noise on Y and Dir)
- Snapshot Ensemble
- Downsample 2017 data (randomly drop 40% of data for each epoch)
- Keep the length of epoch constant (12,000 samples = 1 epoch), regardless of the total number of training samples
    - This helped me estimating running time in 2nd stage

EDIT:
Here is my solution code.
https://www.kaggle.com/nyanpn/pytorch-transformer-public-14th-private-22th
