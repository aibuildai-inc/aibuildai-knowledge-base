# [42nd place solution]

Competition: global-wheat-detection
Rank: #42
Source: https://www.kaggle.com/c/global-wheat-detection/discussion/172409

Congratulation to all, especially to the prize winner and gold medalists and thanks to the organizers for this interesting competition.
So here are some essential points of our solutions (I will update more detail later)
.png?generation=1596588070622865&amp;alt=media)
**1. Validation strategy**
I use a fixed validation set from the beginning of the competition (663 images). The validation data is from usask_1 and a part of other sources.
**2. Augmentation**

 2.1. My custom augmentation: 
    - crop and pad: I realize that the boxes at the border after cropping is not good. Then I remove all the boxes at the border  and remove the image content inside the boxes too.
    - crop and resize
    - resize and pad
    - [Color Transfer between Images](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf)
    - mixup: Instead of mixup 2 images in training set, I use 1 image in training set and 1 image without wheat head. 
    - Rotation: random rotate images and use pseudo label as ground truth.

  2.2. Albumentation:
    - A.HorizontalFlip(p=0.5),
    - A.VerticalFlip(p=0.5),
    - A.RandomRotate90(p=0.5), 
I don't use mosaic augmentation because it does not want to cooperate with my custom augmentation.
**3. SPIKE dataset:** I use WBF on pseudo label and provided ground truth label.
**4. Results**


P/s: Is there a chance for us to move into gold zone after verifying solution?
