# 13th Place Solution

Competition: image-matching-challenge-2024
Rank: #13
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510295

First, I would like to express my gratitude to the competition organizers and Kaggle staff. I have learned a lot from this competition.

# Overview of 13th place solution


Our basic strategy involves using a rotation-resistant model based on [IMC2023 6th place method](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417045), along with a multi-model approach using Alike-LightGlue. Moreover, we made several adjustments depending on the scene.

# Measures for Transparent Objects
## 1. Cropping the Center Portion
Our teammate @sugupoko, discovered that using only the central part of the image improves accuracy for transparent objects. This approach was based on the assumption that transparent objects do not move from the center. As a result, our local score for the cylinder exceeded 0.2. Additionally, by setting the camera model to single-camera, our cylinder score reached **0.463**.



## 2. Removing Keypoints with Close Pixel Coordinates
We observed that the cylinder overreacted to reflected light and tended to match incorrect keypoints due to minimal image variation. Therefore, after matching, we removed keypoints where the x and y coordinate differences were both within 5 pixels, as these were likely incorrect keypoints.


# Measures for Church (symmetries-and-repeats)
The challenge with church was distinguishing between the front and back, causing the camera to focus on the front. 

To capture finer details beyond just the clock, we implemented an approach to detect keypoints by dividing the image into four sections. Ultimately, with a model ensemble and setting the camera model to simple-pinhole, we achieved a score of **0.3561** on the train data.
(Note: Our final submission used simple-radial for all scenes.)
However, this alone did not resolve the issue of distinguishing between the front and back as mentioned above.

# Approaches We Tried but Did Not Work
- I had high expectations for end-to-end matching models. However, due to inference time constraints, I could not include RoMA and OmniGlue in the final submission, and Xfeat did not achieve high scores locally.
- Similar to IMC2023, I implemented an approach to determine and crop the Region of Interest (RoI) based on matched points, but this worsened our scores.
- I tried image correction using CLAHE to deal with dark images, but this also did not work.
