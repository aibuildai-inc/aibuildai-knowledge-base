# 10th MaskRCNN without ensemble and TTA solution.

Competition: airbus-ship-detection
Rank: #10
Source: https://www.kaggle.com/c/airbus-ship-detection/discussion/71607

Congratulations to all the winners ! Since most team used Unet-based solution so I think others may be interested in MaskRCNN based solution.

My pipeline is ordinary: Unet101 with 384*384 + MaskRCNN.
No TTA and ensemble because the score all drop in public LB. (But actually better in Private LB)

## What Works ##
1. I implemented [Online-Hard-Example-Mining][1] selecting top 128 roi's for training.

2. Multi-Scale Training: Since small object are hard for detection, I resize the image randomly from 1200 ~ 2000.
3. I used [softnms][2] for post-processing. I used mask's overlap instead of box overlap to rescore each instance to avoid dealing with the rotated box problem.
4. MaskRCNN is prone to overfitting (The lesson learned from DSB2018). Data augmentation with MotionBlur, GaussNoise, ShiftScale, Rotate, CLAHE, RandomBrightness, RandomContrast ...
5. Enlarge mask crop from 28 to 56, using diceloss + BCE loss.

## Doesn't Work ##
 1.  Add stride 2 featuremap in FPN and add size 16 anchor. Improve local cv to 0.61 but worse public LB and private LB.
 2. [Cascade-RCNN][3] No improvement.
 3. Try to turn mask prediction into box as post-processing.

I have implemented TTA and checkpoint ensemble (see eval.py) but all results in worse public LB. Turns out they are better in private LB (best 0.853 TTA with scale range(1200,  1400 ... 2000)). 
The final score (0.851) are based on single model without TTA. 

Hope other team using Detection-Based solution can share their experience too ! I wonder how rotated box will perform.

The un-cleaned code is [here][4]. (will be cleaned after my recent deadline ...)



  [1]: https://arxiv.org/abs/1604.03540
  [2]: https://arxiv.org/abs/1704.04503
  [3]: https://arxiv.org/abs/1712.00726
  [4]: https://github.com/tkuanlun350/Kaggle_Ship_Detection_2018
