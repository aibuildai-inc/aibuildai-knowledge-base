# [6th place solution] EfficientDet + resnet18

Competition: nfl-impact-detection
Rank: #6
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/208833

Thank you to all participants and organizers!
Thanks especially to @its7171 and @nvnnghia for sharing amazing notebooks!

My solution is made up of the following two stages.
(Stage-1) Detect all helmets with EfficientDet-d5
(Stage-2) Classify all helmet to impact or no impact with resnet18



### Stage-1
I trained EfficientDet-d5 on helmet 1class.
For training, I used the frames where collisions exist and frames where they don't. Training images are extracted from the video every 10 frames and +-5 frames of the frame with collisions.
Images were enlarged to 1280*1280, and only horizontal flip was used for augmentation because loss converged faster.
The inference was performed on two images, the original image, and the horizontally flipped image, and the results were merged using wbf.

### Stage-2
I trained resnet18 on 2 classes(impact or not) and used it to classify the targeting helmet had collided.
For resnet18 input, I cropped the image around the helmet and prepared 9 images from 4 frames before and after.
The cropped image size is three times the longest side of the helmet rectangle(max(w,h) * 3).
The cropped image was padded to square and resized to 112*112.

As in stage 1, the training images were extracted from the video every 10 frames and only the collision frame.
When creating training data, the same evaluation criteria were used to label whether a collision occurred or not(impact=1, confidence>1, visibility>0).
For augmentation, I used the alubumentation library and applied the following
- HorizontalFlip
- RandomBrightness
- RandomContrast
- one of(MotionBlur,MedianBlur,GaussianBlur,GaussNoise)
- HueSaturationValue
- ShiftScaleRotate
- Cutout

In addition to the above, I also used the following augmentation
- Changed crop size randomly (max(w,h) * 2.5 ~ max(w,h) * 3.5)
- Shifted cropping position randomly (-max(w,h) * 0.07 ~ max(w,h) * 0.07)
- Replaced a random frame image with a black image (All 0's numpy array)

When doing augmentation, the same augmentation is applied to all images in 9 frames.

I trained two resnet18 models(2d conv and 3d conv).
For the 2d conv resnet18 input, the 9-frame images are combined in the RGB channels to form a 27-channel image.
For the 3d conv resnet18 input, the 9-frame images are converted to a 9frame movie.

The inference process was performed on the two models, and the final output was the ensemble of the respective outputs.
I also used the same flip tta as in stage-1.

### Post-Processing
If helmets are detected in the same position within 4 frames, only the middle frame will be kept, and the others will be ignored.
In addition, I ignored the first and last 10 frames of the video because it is expected to be a low collision.

### Score
cv for EfficientDet-d5 : 0.926(for all helmets)
roc_auc for 2d conv resnet18 : 0.964
roc_auc for 3d conv resnet18 : 0.967
cv for local : 0.53
public score : 0.56
private score : 0.59
