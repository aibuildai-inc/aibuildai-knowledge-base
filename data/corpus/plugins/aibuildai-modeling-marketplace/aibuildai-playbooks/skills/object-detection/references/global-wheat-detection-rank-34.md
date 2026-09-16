# [34th Place Solution using single EfficientDet + Pseudo Labeling]

Competition: global-wheat-detection
Rank: #34
Source: https://www.kaggle.com/c/global-wheat-detection/discussion/175979

First of all Congratulation to all, especially to the prize winner and gold medalists and thanks to the organizers for this interesting competition!

So here is our solution to this competition (Very very basic):

## Training

- Single EfficientDetD5 (Thanks to @shonenkov for his great notebook)
- 40 epochs using full dataset (No CV)
- Image size: 512 * 512
- Augmentations: Random Crop, Random Cutout, Hue Saturation Value, Brightness Random Brightness Contrast, Random Horizontal and Vertical flips and random Cutmix.
- Training notebook: https://www.kaggle.com/kaushal2896/gwd-efficientdet-training-on-entire-dataset

## Pseudo Labeling

- Pseudo labeled the above model with 10 epochs (Thanks to @nvnnghia for his work)
- Used the same augmentations even while pseudo labeling: Random Crop, Random Cutout, Hue Saturation Value, Brightness Random Brightness Contrast, Random Horizontal and Vertical flips and random Cutmix.
- Same image size: 512 * 512.
- Used both TTA and WBF for final predictions.
- Pseudo Labeling notebook: https://www.kaggle.com/kaushal2896/efficientdet-pseudo-labeling-wbf-tta

## What didn't work for us?

- Custom augmentations like mixup.
- Training on larger image sizes without Random crop augmentation.
- Same solution with EfficientDetD6.
- Training on higher number of epochs.
- FasterRCNN with ResNet51 and ResNet101.


Here are different approaches I've tried during this competition: https://github.com/Kaushal28/Global-Wheat-Detection
