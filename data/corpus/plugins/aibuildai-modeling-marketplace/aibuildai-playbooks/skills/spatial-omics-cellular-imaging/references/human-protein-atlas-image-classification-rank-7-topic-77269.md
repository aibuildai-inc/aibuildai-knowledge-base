# 7th place solution

Competition: human-protein-atlas-image-classification
Rank: #7
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77269

Hello everyone,

Below is the key points of my 7th place solution:

**network input:** RGB with HPA external

**network architecture:** replaced the last global average pooling by concat([attention weighted average pooling, global max pooling, global std pooling]), all the statistics were 2D-plane-wise calculated, followed by two fc layers.

**data augmentation:** contrast(applied independently on each color channel), and rotate, scale, shear, shift

**ensemble:** weighted average of ten models (3 x resnet18 on 1024x1024, 3 x resnet34 on 1024x1024, 2 x resnet34 on 768x768, 2 x inceptionv3 on 768x768)

**a trick:** There are tons of duplicates in the test set. I managed to find some easier duplicates using pair-wise correlation on RGBY separately. Averaging the output probabilities of the duplicates added around 0.04-0.05 LB. I believe the treasure lies in the duplicates that are harder to find because their prob output should differ more.

**thresholds:** I need more still in this. For me, the best seems to be 0.2 with some high occurrence class set to 0.3, but I really believe optimal thresholds depends on the models, and luck. Since we know there are leaks for the rare classes on public LB, I gambled to lower the thresholds of the five rarest classes to 0.1 and 0.05 (got slightly worse public score), used those two as my final two, and in the end the private LB score dropped too.

**Correction:** Averaging the output probabilities of the duplicates boosts 0.004-0.005 LB, not 0.04-0.05
