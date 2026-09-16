# 8th Place Solution

Competition: image-matching-challenge-2024
Rank: #8
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/509902

Thanks to the competition organizers and Kaggle staff for hosting this amazing competition and solid support.
The image matching challenge competition gives us a lot of insight every year. 
I really enjoyed participating.

## Overview of 8th place solution


- Basically, I adopted the ALiked + LightGlue method.
- Since the keypoints of ALiked + LightGlue method decreases with image rotation[1], one image was rotated every 90 degrees to search for corresponding points (1st step).
- Next, using the keypoints obtained in the 1st step, I corrected the orientation of the two images and performed image matching again (2nd step). Affine transformation using HomographyMatrix is ​​used to correct the image orientation. The reason for this is that the orientation can be corrected without specifying the rotation angle between the images.

- In pycolmap's incrementalMapping, this function was implemented under the camera model with "simple-radial" and "simple-pinhole" settings.  
To eliminate randomness in the results, "simple-radial" twice and "simple-pinhole" once were ran.  
From the results of "simple-radial" and "simple-pinhole", the largest model was adopted as the submission.

- Additionally, in order to complete the above process within 9 hours, the notebook had 2 threads (to process keypoints extraction) and 2 forked processes (to process colmap). I used T4x2 notebook and  assigned each GPU to each thread for keypoints extraction.

## Keypoints extraction
A conceptual diagram of keypoints extraction is shown below.


## Reference
- [1] [https://www.kaggle.com/code/motono0223/rotation-effect-for-image-matching](https://www.kaggle.com/code/motono0223/rotation-effect-for-image-matching)
