# 46th solution

Competition: image-matching-challenge-2023
Rank: #42
Source: https://www.kaggle.com/c/image-matching-challenge-2023/discussion/416777

Our method consists of three simple parts: keypoint matching, structure from motion, and post-processing. I will briefly explain each of them with a focus on the differences from the baseline.
## Keypoint Detect and Matching
We adopted a method that performs keypoint extraction and matching separately, rather than an end-to-end matching method that can share 3D model points across many images. Ultimately, we only used KeyNetAffNetHardNet, but if we had more time, we would have liked to ensemble it with SuperPoint-based methods.
By making simple changes listed below, we can improve the score.
- increasing the number of keypoint (2048→2048*4).
- extracting keypoints from different resized images.
- using algorithm adalam and Orinet written in the codes.
- using both the Fundamental matrix and the Homography matrix and merging the two results
for narrowing down the matching based on geometric characteristics.
- setting an upper limit on the number of feature point matches per pair of images for feature point matching to reduce the computational complexity of 3D reconstruction.

## Structure from Motion
We tried multiple minimum matching numbers and used the model that estimated the largest number of images that could be estimated as the final estimation result. However, if the proportion of images that could be estimated exceeded the threshold when trying the final matching number from the largest one, we did not try any minimum matching below it.

## Post-Processing
Images that could not be estimated by colmap were estimated using cv2.solvePnPRansac.
