# 12th place solution - MLP Based (Structured Keypoint Pooling network)

Competition: asl-signs
Rank: #12
Source: https://www.kaggle.com/c/asl-signs/discussion/406300

Thanks to the hosts and the Kaggle team for hosting such an amazing competition.\
Below is my brife solution.

## Model

* MLP based. I used a modified model of [the Structured Keypoint Pooling network](https://arxiv.org/abs/2303.15270).
* Input joint and bone separately. Bidirectional lateral Connection. Concat after 2 mlp blocks. 
  * Refered to [Two-Stream Network for Sign Language Recognition and Translation](https://arxiv.org/abs/2211.01367)

## Data
* Only original data
* Lip + Pose (upper body) + Left hand + Right hand
* Resize to 64 frames, if long.
* Feature: xy (joint or bone), motion of 1&2frame.
* So, input shape is [64(frame), (107(joint) + 142(bone)) x 2 (xy) x 3 (+1&2motion)]
* Exclude data with an low estimation probability on the training data (1%). [CV: -0.006, publicLB +0.01]

## Augmentation
* Flip [CV: +0.02]
* Scale [CV: + 0.008]
* Rotate3d all data [CV: + 0.003]
  * Better than Rotate2d by CV 0.002
* Rotate3d hands individually [CV: +0.002]
* Resize frame (only shorten the frame) [CV: + 0.005]
* Cut the first or last frame [CV: +0.001]
* Swap hand or lip with same sign. [Hand CV: +0.006, Lip CV: + 0.002]

## Loss
* CrossEntropy (label smoothing 0.4)
  
## What Didn't Work
* Word based-label smoothing
* Manifold mixup
* Shift DA
* Adding eye
* Pseudo labels

## Especially thanks to these kernels
* [[LB 0.67] one pytorch transformer solution](https://www.kaggle.com/code/hengck23/lb-0-67-one-pytorch-transformer-solution) [@hengck23](https://www.kaggle.com/hengck23)
* [GISLR [LB 0.63]: On the Shoulders](https://www.kaggle.com/code/roberthatch/gislr-lb-0-63-on-the-shoulders) [@roberthatch](https://www.kaggle.com/roberthatch)
* [Animated Data Visualization](https://www.kaggle.com/code/danielpeshkov/animated-data-visualization) [@danielpeshkov](https://www.kaggle.com/danielpeshkov)
