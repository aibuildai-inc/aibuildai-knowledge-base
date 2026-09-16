# 1st place solution (1/26 details updated)

Competition: pku-autonomous-driving
Rank: #1
Source: https://www.kaggle.com/c/pku-autonomous-driving/discussion/127037

Thanks to everyone and congratulations to all the top teams. I'm on vacation, so I will post the details after the Chinese New Year.

# In brief
 - data augmentation: h-flip, 3 axis rotate, color, noise, blur.
 - [keras hourglass centernet](https://github.com/see--/keras-centernet)
 - perspective transform for efficiency
 - regression of yaw, cos(pitch), sin(pitch), rot_pi(roll), x, y, z, r
 - blend 6 results (2 types of head * 3 types of transform)
 - post process by fitting LB


# Especially thanks to these kernels:
 - [CenterNet Baseline](https://www.kaggle.com/hocop1/centernet-baseline) @hocop1
 - [Augmented Reality](https://www.kaggle.com/ebouteillon/augmented-reality) @ebouteillon
 - [metrics evaluation script](https://www.kaggle.com/its7171/metrics-evaluation-script) @its7171


Happy Chinese New Year

# 1/26 details updated

# Network

[network]

My approach is based on [keras hourglass centernet](https://github.com/see--/keras-centernet).
Some notes:
- ***6 Dof***: regression of yaw, cos(pitch), sin(pitch), rot_pi(roll), x, y, z, distance
- discard ***XY bias*** result finally.
- remove ***Car types*** and ***XY bias*** in my second model.

**# Perspective transform**
Two purpose:
- reduce the size gap between small(far) and large(near) cars.
- cover more outliers without extending image.

I find the model don't predict well on the large car when I increase the input size, so I make them smaller. The extra benefit is to enclose outliers.

**Original image:**
[original]
**Transformed:**
[transformed]
**Coverage comparison:** (dots denote the GT location)
[coverage]
**Validation result with outlier:** (red dot: GT, green: predict heat map)
[transformed]

**# Coordinate reference**
I think the same feature in different locations should get different results. So I join this layer to get better predictions, and apply random crop when training.

# Data augmentation
I use: h-flip, camera rotate, color, noise, blur.

**# Camera rotation**
This is the most important part of my approach. Since I only have 4001 training images, 5 bad, and 256 for validation. It is easily to overfit without rotation augmentation. 

**The augmentation looks like:** (center is original image)
[rotation augmentation]
Please refer to [this kernel](https://www.kaggle.com/outrunner/rotation-augmentation) for details.

# Training
- Focal loss for heat map
- Huber for regression
- Adam optimizer
- Manually adjust learning rate from 10^-3.5 to 10^-5.5
- About 0.4M iterations
- Train: full network -&gt; part of -&gt; head only -&gt; full ...
- Change input size (random corp) and batch size every iteration
- One 2080Ti per training. (I have 2)
- Total 6 models, 2 heads * 3 transforms (different parameter and input size)

# Inference
Please refer to [this kernel](https://www.kaggle.com/outrunner/autonomous-driving-1st-place-solution-inference) for details.

**# Test time augmentation**
Flip and multiple transformations, weighted average the predictions.

**# Blending**
Transform predictions to one model's transformation, weighted average 6 results.

**# Weighted average neighborhood**
When decoding, not only use the local maximum point but also take into account
the prediction around it.

# Metric probing and Post processing
This is the first time I join a competition without knowing the evaluation metric. The probing is interesting, but there are some weird characters in the metric.

**Probing procedure:**
[Probing process]
**# Image wise**
Split test images to two sets A and B, then: **score(A) + score(B) = score(A+B)**

**# Confidence independent**
So the metric is something like **F1** or **TP/(TP+FN+FP)**

**# Rotation**
θ and θ+2π differently, so the score is directly impacted by roll prediction. Therefore, I train a model to predict global roll and get the score improvement.

**# Translation**
The most weird thing is that when I shift X by some pixels, the LB score change  significantly. So I guess the Metric is:
```sh
(abs(x-xp)/abs(x) + abs(y-yp)/abs(y) + abs(z-zp)/abs(z))/3
```
**And increase the threshold when abs(x) is small:**
[Confidence threshold]

**The overall post-processing:**

- [opt] replace x, y by X, Y, z, r (just like everybody do)
- [roll] replace instance roll by global roll
- [xs] shift X 2 pixels (don't know why)
- [rx] drop some cars whose x are near by zero, and keep a least car number per image
- [dz] drop duplicate cars

**# table of results:**
[table of results]
*# parameters are the same as final submission, and some procedures are dependent*
