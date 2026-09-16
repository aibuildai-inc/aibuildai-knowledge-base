# 17th place solution - Upscaling & Models

Competition: image-matching-challenge-2022
Rank: #17
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328803

Our team would like to thank Google Research and Kaggle for organizing such an amazing competition.

# Models

Our final solution which scored 0.835 consists of: 
- **LoFTR DS/OT** (default settings for DS, 0.55 match_coarse threshold for OT)
- **SuperPoint/SuperGlue**
- **DKM** (100 features)
- **MatchFormer-LargeLA** (0.55 match_coarse threshold)

# Upscaling with Lanczos interpolation over 8×8 pixel neighborhood

Part which had the biggest impact on our solution was **upscaling** with Lanczos interpolation. Generally speaking, we upscaled images for all of our models except MatchFormer by a factor of **1.5x**, and then made them divisible by 8. In the end, position of keypoints was adjusted using the same scale factor. We think that CNNs used in many models, especially in LoFTR, can simply extract richer local features from upscaled images.
> img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)
> 
> w, h = img.shape[1], img.shape[0]
> w, h = w // 8 * 8, h // 8 * 8
>
> img = img[:h, :w, :]

# Outlier detection

For outlier detection we incorporated simple MAGSAC from OpenCV:
> cv2.findFundamentalMat(keypoints_1, keypoints_2, cv2.USAC_MAGSAC, ransacReprojThreshold=0.25, confidence=0.99999, maxIters=100000)

# What did not work for us?

Here's a list of things which unfortunately didn't improve our score.

- SuperGlue re-training or fine-tuning (we did this to combine it with DISK descriptors)
- DISK + OANet
- Calculating additional keypoints from rotated images
- Inverse image order for LoFTR
