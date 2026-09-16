# 9th Place Solution

Competition: pku-autonomous-driving
Rank: #9
Source: https://www.kaggle.com/c/pku-autonomous-driving/discussion/127052

I thank the host Peking University/Baidu and Kaggle team for holding this attractive competition, and congrats to all prize and medal winners.
Here is a brief summary of my solution (under construction).

# Approach
I used two independent models. The first model (Model A) detects cars and estimates their orientations. The second model (Model B) estimates depth map of each image. Using image coordinate (ix, iy) from Model A, depth (z) from Model B, and camera intrinsic, 3D coordinate (x, y, z) of each car is calculated. Estimating accurate depth is harder than car detection or orientation estimation. Thus I separated depth part to different dedicated model (Model B).




# Model A
Similar to CenterNet, but there are some modifications in targets and losses.

## Targets
Model A detects cars, estimates their image coordinate (ix, iy)(not 3D camera coordinate (x, y, z) required for submission), and (yaw, pitch, roll).
Therefore, the targets are (conf, dx, dy, sin(pitch), cos(pitch), yaw, roll).
conf is confidence map for detecting car centers. (dx, dy) is relative position of car center in feature grid (0-1).
(yaw, pitch, roll) is local orientation, not the orientation in camera coordinate system that is given as groudtruth. I used local orientation because original ground-truth orientation is hard to estimate from car appearance without context (camera position and image coordinate).





[1] A. Mousavian et al., "3D Bounding Box Estimation Using Deep Learning and Geometry," in Proc. of CVPR, 2017.

This modification of orientation is done by rotation matrix that moves camera center to target car center:

```
yaw = 0
pitch = -np.arctan(x / z)
roll = np.arctan(y / z)
r = Rotation.from_euler("xyz", (roll, pitch, yaw))
```

## Losses
Cross entropy is used for conf, and L2 loss is used for the other targets.

## Architectures
I separately trained different models as Model A (four models) to detect different sizes of cars by grouping cars according to their depth; 0-25, 20-50, 40-80, and 70-180. For the former two groups (closer cars), efficientnet or se-resnext is used to get x32 downsampled feature map. For the latter two groups, segmentation models (efficientnet or se-resnext + FPN) are used from segmentation_models.pytorch to get finer feature maps (x16).

## Augmentations
Augmentation is difficult part; it is related to how ground-truth of zoomed or flipped test images is created.

- flip (around principal point instead of image center: img[:, :3374] = cv2.flip(img[:, :3374], 1))
- rotation (around principal point; img = np.array(Image.fromarray(img).rotate(theta, center=(1686.2379, 1354.9849))) (-7 to 1 degrees)
- RandomBrightnessContrast
- Random scaling and crop

## Optimizers
Trained for 160 epochs with Adam; LR = 0.0001 and decreased by 0.1 at epoch 100 and 140.

## Ensemble
Several k-fold models are integrated at feature map level (model raw outputs are averaged), but it seems to not work (why...?)

# Model B
## Input
For Model B, fixed area of input images are used for training and test: img[1558:, 23:3351]. Also, relative image coordinates from principal point is added to input in order to exploit the context of fixed camera position against ground (thus, input is gray image, dx, dy).

## Targets and Losses
The second model predicts only the depth of cars. Actually, I trained Model B to predict (x, y, z) but used only z information.
Thus, target is 3D car coordinate (x, y, z). The loss is calculated only from the pixels of feature map that car centers exist.
Loss function used here is MSE normalized by the distance from camera: ||pred - gt||_2 / ||gt||_2.
This selection comes from my assumption that the evaluation criterion about translation is relative rather than meter.

## Architectures
Segmentation models (efficientnet or se-resnext + FPN) are used get  feature maps (x16).

## Augmentations

- flip (around principal point instead of image center: img[:, :3374] = cv2.flip(img[:, :3374], 1))
- RandomBrightnessContrast

## Optimizers
Trained for 100 epochs with Adam; LR = 0.0001 and decreased by 0.1 at epoch 70.

# Questions
After finishing the competition, several questions are still left to the participants. I really appreciate if the organizers answer to the following questions.

- How was ground-truth for flipped test image created? I guess simply done by x = -x, roll = -roll, pitch = -pitch and this is not accurate as discussed in https://www.kaggle.com/c/pku-autonomous-driving/discussion/123653
- How was ground-truth for zoomed test image created? I guess it is done by z = z / scale_factor. The other option is leave the ground truth as it is but I think this is less appropriate as we do not know new camera intrinsic.
- What was evaluation metric? Discussed in https://www.kaggle.com/c/pku-autonomous-driving/discussion/124489
