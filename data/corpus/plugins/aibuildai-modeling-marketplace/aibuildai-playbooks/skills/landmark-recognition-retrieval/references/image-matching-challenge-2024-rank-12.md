# 12th Place Solution

Competition: image-matching-challenge-2024
Rank: #12
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510673

# Acknowledgments
First of all, I'd like to express my gratitude to all the hosts and Kaggle staff for organizing such an interesting and challenging competition!

# Overview
* Basically, I adopted the same as Baseline for getting image pairs, ALIKED for keypoints detection, and LightGlue for matcher.
* When detecting keypoints on the image, the image was divided into 4 images[1] and resized to a size that improved both CV and LB for transparent and non-transparent objects.
* When matching keypoints, one image is rotated 0°, 90°, 180°, 270°, and the image is matched in 4 divisions with the pair with the highest number of matches.
* The matcher_th and min_matches for matching between the 4 images were adjusted to improve CV and LB for both transparent and non-transparent objects.

# Pipeline
1. Whether a dataset is a transparent object or not is determined by setting the threshold=0.7 for the average normalized correlation coefficient between the average image of all images in a dataset and each image.
(The threshold was adjusted in the provided data experimentally.)

2. Splits the image into 4 images, resized to a size that improves the scores for transparent(=1280) and non-transparent(=1600) objects, and detects keypoints. 
3. Select the one with the highest number of matches while rotating 0°, 90°, 180°, 270°, and select the one with the highest number of matches.
4. Keypoints were matched by setting matcher=0.27 and min_matches=30 for transparent objects and matcher=0.7 and min_matches=28 for non-transparent objects.



# Reference
* [1] [https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/416918](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/416918)
