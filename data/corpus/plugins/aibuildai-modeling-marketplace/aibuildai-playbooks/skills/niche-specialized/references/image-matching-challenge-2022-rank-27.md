# 27th place 0.823 public lb - 0.828 private lb

Competition: image-matching-challenge-2022
Rank: #27
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328888

Thanks to the organizers for hosting this competition and congrats to all the participants.

**Summary**

My final solution is a combination of several models:

1. 3 x LoFTR (472x840, 704x1280, 936x1704)
2. SuperGlue + SuperPoint (472x854, max_keypoints 4096)
3. DKM (472x854) with 600 feature points

Before merging all the keypoints for finding fundamental matrix I prefiltered DKM outliers by applying DEGENSAC with threshold 0.5. 

Final RANSAC parameters:

  `cv2.findFundamentalMat(keypoints_1, keypoints_2, cv2.USAC_MAGSAC, ransacReprojThreshold=0.25, confidence=0.99999, maxIters=50000)`

**Tricks that made a huge impact on the score:**
1. Upscaling images for LoFTR and making them divisible by 8
2. Changing DKM default image resolution
3. Filtering out DKM outliers before the final fundamental matrix estimation
4. RANSAC parameters tuning
5. Ensembling

**Things that didn't work for me:**
1. HardNet, Hynet, SosNet
2. R2D2, DISK
3. OANet
4. SGMNet (https://github.com/vdvchen/SGMNet)
5. LISRD (https://github.com/rpautrat/LISRD)
6. Patch2pix (https://github.com/GrumpyZhou/patch2pix)
7. Segmenting out shaky classes (people, cars, sky etc.) with models pretrained on MIT ADE20K scene parsing dataset
(https://github.com/CSAILVision/semantic-segmentation-pytorch)
8. Other types of RANSAC (GC-RANSAC, DEGENSAC etc.)
