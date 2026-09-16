# 26th Place Solution

Competition: image-matching-challenge-2024
Rank: #26
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/509918

First and foremost, I would like to thank the competition organizers, and the Kaggle team for providing a platform, and the right atmosphere to build up our knowledge and be part of this great community.

1. Introduction
This Kaggle competition focused on the challenging task of estimating the camera rotations and translations from a collection of images. The proposed approach begins by extracting features from the input images, followed by comparing and matching the keypoints across the images. A 3D reconstruction program, in this case, PyColmap, is then utilized to determine the camera positions that captured the images.

2. Approach
The approach employed in this work involves an ensemble of five feature detection and matching models. By combining the keypoints and matches from these models, a more robust set of correspondences can be obtained. The fundamental matrix is then calculated based on these correspondences and directly passed to the 3D reconstruction algorithm, bypassing the exhaustive matching step.



3. Model Performance
The best-performing model achieved the following results:

| Code | PUBLIC SCORE | PRIVATE SCORE |
|------|--------------|---------------|
| [imc24-kornia-work](https://www.kaggle.com/code/nartaa/imc24-kornia-work) | 0.164598 | 0.173382 |

4. References
[imc2023-final-pub](https://www.kaggle.com/code/maxchen303/imc2023-final-pub)
