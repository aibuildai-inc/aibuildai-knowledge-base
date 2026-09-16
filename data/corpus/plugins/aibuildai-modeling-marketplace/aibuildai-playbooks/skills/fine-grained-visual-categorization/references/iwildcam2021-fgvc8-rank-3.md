# 3rd Place Solution

Competition: iwildcam2021-fgvc8
Rank: #3
Source: https://www.kaggle.com/c/iwildcam2021-fgvc8/discussion/244950

For this solution I was based on some of the ideas of the previous year's competitors, so I hope that the description of my solution will also be useful for someone else.

# Dataset
I used the images from the WCS collection and those from the iNat 2017/2018/2021 collections for the shared classes, and I applied [CLAHE & Simple WB](https://www.kaggle.com/seriousran/image-pre-processing-for-iwild-2020) to this whole set to enhance the nighttime images.

I applied Megadetector V4 inference on this set of images (including those of WCS labeled as *empty*) and I considered only detections with score ≥ 0.3 for WCS and those with detection label *Animal* and score ≥ 0.95 for iNat collections. Then, I used a square crop around the detection bbox, with the size equal to the largest side of the bbox, trying to keep the animal centered in the square, unless it was outside the image boundaries. I assigned to each of these crops the image-level annotation label (the species or *empty*) from the original full image, in order to create a *train bbox dataset*, that I split in a stratified fashion (train 80% - validation 20%), grouping the WCS images by location.

# Training
I fine-tuned three models with pre-trained ImageNet weights, and the train was done in different image resolutions: ResNet152 (224 px), EfficientNetB3 (300 px) and EfficientNetB7 (600 px).
During the training very simple data augmentation was performed: random rotation, random translation, random horizontal flip and random contrast.

## Geo-prior model
I used the sin/cos representations of location and time-of-year information from the WCS/iNats sets to train a [Geo-prior model](https://arxiv.org/abs/1906.05272), which I used as complementary a priori information when performing model inference on the WCS test data. For this, I used the [TF implementation of the Geo-prior model training](https://github.com/alcunha/geo_prior_tf).

# Predictions on Test data
For the WCS test data I followed a similar methodology as above (CLAHE & Simple WB, Megadetector V4, detections with score ≥ 0.3 and square image crops) to create a *test bbox dataset*.

## Movement detection
One of the main drawbacks of classification pipelines based on Megadetector predictions is the large number of false detections it produces, which are sometimes difficult to be eliminated. One way to do this when the temporal and location information is available is to use some motion detection scheme.
In my experiments, I tested the technique called Accumulated averaging to find those regions in the images where an animal was detected but that did not show any apparent change in time (movement). Thus, if a detection has a score < 0.9 and in the bbox region no movement was detected through the frames of the sequence, that detection was marked as invalid. For this, I have been inspired by the ideas of the [MotionMeerkat project](http://benweinstein.weebly.com/motionmeerkat.html) and I have used pieces of their [code](https://github.com/bw4sz/OpenCV_HummingbirdsMotion/blob/master/MotionMeerkat/BackgroundSubtractor.py#L28) and their parameter values.
There are other methods to detect motion that seem to be more effective (e.g. Gaussian Mixture-based B/F Segmentation) but I did not test them in these experiments.

Below are two examples of sequences with detections that were correctly discarded (reds) and others that were correctly accepted (greens) by this method:




## Model inference and averaging
I applied the inference of the three classification models separately on the *test bbox dataset*, multiplying the output vector of the classifier by the output vector of the Geo-prior model for each image, and then performing a weighted average of the predictions of the three models. The weights of each model were based on the results of the evaluation on the validation set.

## Creation of the submission file
To determine the species and the number of individuals present in each sequence I did the following:
- I assumed that there was only one species present in each sequence and this was determined from the final output of the weighted classifiers/Geo-prior on the valid detections (not discarded by the motion detection method and not classified as *empty*) that had a *classification probability* ≥ 0.5, taking as the *species of the sequence* the one that is repeated the most (mode) in the whole sequence.
- The number of individuals is calculated simply by taking the maximum number of valid detections present in any of the images in the sequence.

## Additional experiment
Inspired by the [ideas of the winning team of last year's competition](https://www.kaggle.com/c/iwildcam-2020-fgvc7/discussion/158370) I did the following:
* I used the original *full images* of the WCS train set to fine-tuned two models with pre-trained ImageNet weights: EfficientNetB3 (res 300 px) and EfficientNetB7 (res 600 px), and I did a weighted average of both model predictions of the full images of the WCS test set (also using the Geo-prior model).
* I did a weighted average of the predictions of the EfficientNetB3 and EfficientNetB7 *bbox models* described above (I discarded ResNet152 predictions for this experiment) on the *test bbox dataset*.
* Then, for the images that had at least one valid detection I did a weighted average of these two averaged predictions (`0.3 full image + 0.7 bbox`), and for the images that did not have valid detections I used only the full image model.

This final predictions outperformed the public and private scores of all my previous submissions, but as it was a late submission it was not reflected in the LBs.
