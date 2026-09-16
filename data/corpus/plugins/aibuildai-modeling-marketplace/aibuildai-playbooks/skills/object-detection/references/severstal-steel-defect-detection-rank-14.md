# 14th Place Solution

Competition: severstal-steel-defect-detection
Rank: #14
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114367

This time, I was trying to be more methodological instead of randomly picking up ideas from papers. I spent more time thinking and digging into why my current model was doing bad before training a new model. Plus luck, I finally got my first gold.

# Data augmentation and pre-processing
I used `albumentation` to do data augmentation
```
def aug_medium(prob=1):
 return aug.Compose([
     aug.Flip(),
     aug.OneOf([
            aug.CLAHE(clip_limit=2, p=.5),
            aug.IAASharpen(p=.25),
            ], p=0.35),
     aug.OneOf([
         aug.RandomContrast(),
         aug.RandomGamma(),
         aug.RandomBrightness(),
         ], p=0.3),
     aug.OneOf([
         aug.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
         aug.GridDistortion(),
         aug.OpticalDistortion(distort_limit=2, shift_limit=0.5),
         ], p=0.3),
     aug.ShiftScaleRotate(rotate_limit=12),
     aug.OneOf([
            aug.GaussNoise(p=.35),
            SaltPepperNoise(level_limit=0.0002, p=.7),
            aug.ISONoise(p=.7),
            ], p=.5),
     aug.Cutout(num_holes=3, p=.25),
 ], p=prob)
```
Then I took 256x512 crops from the augmented image. Note that many images contain large black area, and there is always steel part shown on at least one end of an image. Simple random crops would make training inefficient, so here is what I did:
1. Pick a random crop
2. If the proportion of pixels with values &lt; 10 is greater than 85% or the average pixel values of the crop &lt; 15, pick the crop on either left or right end of image depending on which end satisfying the criterion. If nothing works, pick the crop on the right end

# Balanced Sampling
To fight against imbalanced classes, I used balanced sampling. I don't like the approach that random picking defect type (or non-defect) under the same probability. It would make the concept of epoch arbitrary. I prefer deterministic approach, so in each epoch, I did something like
`0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 ...` till every single non-defect (class 0) was sampled once.

# Model
## Model 1:
End-to-end classification + segmentation asymmetric U-Net on training set only.
Backbone: se-resnext50 32x4d
Decoder: CBAM attention and hyper-columns. I thought that the last decoder (and maybe the upsample in the second last decoder as well) was kind of redundant, so I removed it. In this way, the output stride of my model became 2. This saved a lot of my GPU memory. 



I used a similar setup in TGS salt competition too. Note that, unlike other classes, I fed all images to the defect 3 branch instead of defect 3 images only.

## Model 2:
Same as model 1, except using deep-stem and replacing every ReLU with Mish. Trained on training set and pseudo-labels on public test set

# Training
Loss: BCE loss for classification, symmetric Lovasz-Hinge for segmentation
Optimizer: Adam
Scheduler: Warmup+Flat+Cosine. 0.5 epoch linear warmup, 49.5 epochs flat at 1e-4, and 50 epochs cosine to 0
Batch size: 6
No fine-tuning on full images

# Post-processing
Thresholds: since I was using hinge loss, I didn't tune segmentation thresholds
Classification: [0.55, 0.99, 0.25, 0.5]
Minimum pixels: [0, 0, 1200, 0]
These values were determined by one fold of model 1. 
After the competition ended, I tried 0.5 for classification and 0 for minimum pixels for all four defects. It gave me 0.90625 on private test.

TTA: original, hflip and vflip

The masks are actually polygonal bounding boxes. It means that no holes in each part of defect. I randomly saw [this page ](https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html). I decided to gave **Edge-based segmentation** in that page a try. This post-processing consistently gave me 0.00005-0.0001 boost on local validation, public test and private test.

# Pseudo-labelling
Obtained pseudo-labels from 3 folds ensemble of model 1. Removed defect 1 with classification output &lt; 0.85 and defect 3 with classification output &lt; 0.75. I kept the amount of test images around 35% train images for each epoch.

# Ensemble
2 folds of model 1 and 1 fold of model 2. Simple average before thresholding.
My best single fold of model 1 gave me 0.90310 on private test.

# Things didn't work
See [this post](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114199#657496)


Congratulations to all winners! P.S. I am not releasing my code at this moment since the efficiency prize is still ongoing.
