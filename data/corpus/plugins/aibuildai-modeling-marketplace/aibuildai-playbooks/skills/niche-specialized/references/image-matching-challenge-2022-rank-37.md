# 37th place solution

Competition: image-matching-challenge-2022
Rank: #37
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/329012

First of all many thanks to the competition hosts and Kaggle for what was in my opinion the most exciting competition of this year.

I personally started this competition with the only objective being 'to learn a bit more'….that turned out to be a lot more ;-)

With a lot of new techniques, learned models, some nice notebooks and a completely unexpected 37th place I couldn't be more happy.

I started out with following the many wonderfull posts and shared papers posted by 'old-ufo'. Many thanks! They were a terrific starting point.

I used the training data locally to be able todo many validation runs. For determining the effects of any change I made I looked both at local validation score and the effect on the LB as the public part was 51% of the data.

**My final solution:**
	• An ensemble combined of 3 pretrained models.
	• LoFTR. A fixed resize of the longest side to 840 pixels like in the training set. Resize based on INTER_LANCZOS4 interpolation. This boosted my score already very early to 0.807.
	• SuperGlue with a fixed 1024 keypoints. No modified image processing.
	• DKM with a fixed 256 keypoints. Image preprocessing. Resize based on image orientation to sides of 512 and 768 pixels. This boosted my score over 0.820.
	• MAGSAC++ with a threshold of 0.20 and 60K iterations..

I discovered that the resizing of the images for DKM had a very large impact...unfortunately I only had 5 days left to be able to further use the impact of that.

It turned out that my best private LB score is 0.822. In this version I increased the amount of keypoints for DKM to 336 and decreased the SuperGlue amount to 832. However both public LB and local validation score decreased for that version…so unfortunately it was no obvious one to choose.

**What didn't work for me:**
	• I tried ANMS - SSC for non maximum suppression to reduce keypoints. Spend a few evenings on it but couldn't get it to work properly. Interresting to read was that the 10th place solution was able to use it. I will have to revisit that again.
	• I tried dynamically balancing keypoint amounts. If one model finds more keypoints..increase also for other models. And vice versa. The effects seemed to be to little and fluctuated (sometimes better, sometimes worse) compared to fixed amount of keypoints.
	• I tried combining my final ensemble with ASLFeat. It didn't gave me a boost.
	• I tried combining my final ensemble with MatchFormer. It didn't gave me a boost.
	• Using some of the filtered keypoints from SOLD2…again score decreased.
	• Way to many configuration changes for all 3 models that seemed to have mixed and minor effects.
	• Training on the available training dataset. This just took way to long and did only seem to hurt performance.
	• 2 stages with RANSAC. First a quick round to get some more filtered keypoints and in a final stage the long run with many iterations. The score decreased. I read in some of the other solutions that they used something similar were the threshold for the first stage was increased. Very interresting.
	• Different versions of the RANSAC family. MAGSAC++ is just the current best.
