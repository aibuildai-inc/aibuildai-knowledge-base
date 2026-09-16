# Solution description (3rd place)

Competition: inclusive-images-challenge
Rank: #4
Source: https://www.kaggle.com/c/inclusive-images-challenge/discussion/71433

## Some thoughts ##
1) After some analysis of data we noticed that most of differences between train and test (stage 1) was more due to different labelling distributions, than to different geo locations. For example: average number of labels per image in train was 4.083 (maximum labels: 127) while in test only 2.386 (maximum labels: 9). So by assuming stage 1 and stage 2 have similar labeling distributions, we can use very similar, but less risky (potentially less overfitting), thresholds for stage 2 as in stage 1.

2) Since we only have one model to choose in stage 2, in order to balance risk v.s. reward, we chose a slightly more conservative approach than our best LB model in stage 1 – our final selected model scored around 10th position on stage 1 LB, which has a slightly more conservative threshold optimization.

## Models and scores ##
Our solution consists of 7 models: *ResNet50*, *Xception*, *Inception ResNet v2* x 5. All neural nets had the same final Dense (FC) layer with ~7K neurons for all available classes with sigmoid activation. However, the five inception-resnet have slightly different training and image augmentation approaches. All nets had almost the same score ~0.5 at stage 1 after threshold tuning. The best one was Inception resnet v2. ResNet50 was trained on 336x336 resolution, others on 299 resolution. I believe single model is enough here to get almost the same score (late submission for random single model gives: 0.337). For example ensemble with majority voting on stage 1: 0.505 + 0.501 + 0.511 LB: 0.516

## Train/validation split ##

- We used all labels for training: human and computer generated.
- We also used probabilities from computer generated labels rather than fixed 1.0 value.
- We didn't use boxes
- Split was non-uniform stratified by classes (see code below):

<pre>    if count &lt; 50:
        part = count // 10
    elif count &lt; 100:
        part = count // 20
    elif count &lt; 1000:
        part = count // 50
    elif count &lt; 10000:
        part = count // 100
    elif count &lt; 100000:
        part = count // 500
    else:
        part = count // 1000
</pre>

 - Train: 1728299 images Valid: 14743 images

## Training process ##

- Nothing really special: random samples from training images
- Training from scratch (not using ImageNet pretrained weights) due to competition rules.
- We used default binary_crossentropy loss for this competition
- For validation during training we used tuning labels from stage_1 and calculate F2 score on each epoch with different thresholds (THRs). At the final epoch for inception resnet v2:
<pre>    loss: 0.0019 - f2beta_loss: -5.6904e-01 - fbeta: 0.7356
    val_loss: 0.0027 - val_f2beta_loss: -5.6610e-01 - val_fbeta: 0.7038 
    F2Beta score tuning labels: 0.373804 Optimal THR: 0.7
</pre>
- F2Beta score on tuning labels keep improving with improvement of validation, so it mostly used for checking rather than early stopping.
- Local score on tuning labels was almost the same as on leaderboard.
- Heavy augmentations. Very useful library [albumentations][1] helped us here:
<pre>    def strong_aug(p=.5):
        return Compose([
            HorizontalFlip(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=3, p=.1),
                Blur(blur_limit=3, p=.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.1),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
            ToGray(p=0.05),
            JpegCompression(p=0.2, quality_lower=55, quality_upper=99),
            ElasticTransform(p=0.1),
        ], p=p)
</pre>
- One epoch 40000 images. To train model we used 200-500 epochs. Training further continue improving model, but with very small effect.
- For xception and 3 more inception-resnet, we use a different image augmentation strategy. For example, one inception-resnet has image augmentation like this:

<pre>gen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, width_shift_range = 0.1, height_shift_range = 0.1, channel_shift_range=0.1, shear_range = 0.1, zoom_range = 0.1,  rotation_range = 10, preprocessing_function=inception_resnet_preprocess_input)</pre>

## Threshold tuning ##
It was the most important part of the competition. We needed to output discrete set of classes for each image based on probabilities from neural networks. We had wide range of strategies to find optimal thresholds and the F2Beta metric is very sensitive. For example:

 - F2Beta on validation images was about: ~0.73 
 - F2Beta on tuning labels with single threshold: ~0.35 
 - LB Score F2Beta with optimized thresholds: ~0.5 LB score

Optimization process was made on tuning labels. It had 4 parameters:

 - minimum probability for search [MinPS]: 0.01
 - maximum probability for search [MaxPS]: 0.99
 - default probability: 0.99 - We use this probability if class had no entry in tuning labels. Actually, from 7K classes only ~500 had entry in tuning labels. Default prob 0.99 means that model must be &gt;99% sure to use this class.
 - minimum number of entries in class: 1

For conservative models in our final submission we used [0.1-0.9] ranges with 2-3 minimum number entries in class and lower default probability (0.8, 0.9). While on Stage 1 LB it gives worse (less overfitting) result.
Start from setting all thresholds to 0.5 then iterate all over the classes and find threshold for each class in range [MinPS; MaxPS] which maximize the F2Beta score. Do 2-3 overall iterations until F2Beta score stops increasing.

## Ensembles ##
We used simple majority voting for ensemble. If at least 4 (out of 7) models predicted a class then this class goes to the submission file.

## Post processing ##
For F2Beta metric it's better to output something if model didn't predict any class. So rows which have empty labels, we will use UNION across all 7 models to regenerate labels. If all 7 models have no predictions, we then will output 3 most common classes found in training set.

## Failed experiments ##

 - We also tried ResNet152 on 448x448 images, but it gave worse results.
 - MobileNet on 128x128 images - just to check if it could give comparable result, but it didn't
 - SE-ResNext101 for Keras from this repo: https://github.com/titu1994/keras-squeeze-excite-network - very very slow and almost not converges comparing to other models.
 - Label selection/filtering based on “class activation maps” (size of activation field, location, etc.)


  [1]: https://github.com/albu/albumentations
