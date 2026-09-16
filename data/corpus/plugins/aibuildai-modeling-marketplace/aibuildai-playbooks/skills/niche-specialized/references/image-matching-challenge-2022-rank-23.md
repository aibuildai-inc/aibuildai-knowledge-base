# 23rd place solution

Competition: image-matching-challenge-2022
Rank: #23
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/329002

Thanks for the challenge! It was a great learning experience.
Here's a short summary of our best scoring method and some insights and open questions.

1. Upscale image.
2. Use **LoFTR**.
2. If < 2000 matches, use **SE2-LoFTR** but with the two images in reverse order and add the found matches to the earlier ones.
3. If still < 2000 matches, use **DKM** to find ROIs in the images as follows. 
    1. Sample 500 DKM matches. 
    2. Cluster the DKM correspondences (as 4D points) using DBSCAN and crop to the different clusters in both images. 
    3. Run SE2-LoFTR on each of the cropped (and resized) image pairs and add the found matches to the earlier ones.
4. Downscale keypoints to compensate for 1.
4. Run `cv2.findFundamentalMat`.

**SE2-LoFTR**
In our experiments the "big" version of SE2-LoFTR usually slightly outperformed LoFTR while the "small" versions did worse. The disadvantage of the "big" version is that is slower than LoFTR, but we still used it for performance reasons. Using two different networks in stage 2. and 3. improves robustness, but perhaps a retrained LoFTR version could have been an alternative to SE2-LoFTR. Upright images of static objects (buildings) is not a scenario where SE2-LoFTR has a clear upside on LoFTR. Still, I wanted to use SE2-LoFTR as it is my method. ;)

**Adaptively choosing ensemble**
Most teams seem to have ensembled by always using a certain set of methods on all image pairs. We saw on phototourism that when LoFTR finds many matches, the score is usually very good. So we only use additional methods when LoFTR fails to find many matches, saving computational resources in this way.

**DKM**
Simply applying DKM as is yielded a worse score than LoFTR, possibly due to poorer keypoint localizations (this seems to have been solved by some teams by changing the image size used internally in DKM). However DKM has the advantage of always producing as many matches as one wants and usually good ones. The idea of using DKM for finding ROIs in the images is similar to the 10th place solution and could probably be improved a lot by digging into the DKM method.

**Resizings**
Upscaling the input images to longest side 1176 was the single most important reason for the improvement over the baseline LoFTR. Using dimension divisible by 8 is very important for LoFTR due to the 1/8 coarse scale. Modifying LoFTR to use a finer coarse scale and retraining would probably have worked quite well.

**RANSAC versions**
`cv2.USAC_MAGSAC` always hugely outperformed other open-cv RANSAC presets in our experiments. The reason for this is a bit unclear. The last 2 days we tried changing the parameters of MAGSAC (i.e. nbr of local optimization samples and l.o. iterations), which seemed to give improvements on phototourism data using only LoFTR, but this did not translate to the test set and the complete method in the few submissions we had left.

In the end, the big thing we missed was perhaps using the QuadTreeAttention version of LoFTR as described in for instance the 6th, 11th and 21st place solutions. We also did not use SuperPoint/SuperGlue as it was semi-banned. Reading through other solutions we could probably have done some multithreading tricks to be able to get more computations done, we had quite frequent timeout issues when trying things.
