# 4th Place Solution: SIFT + Siamese

Competition: humpback-whale-identification
Rank: #4
Source: https://www.kaggle.com/c/humpback-whale-identification/discussion/82356#latest-496460

My goal in this competition was to learn more about low-shot learning problems and to try to get to GM, so I’ll share what I learned.

I find it useful to try to think like the sponsor and ask why they would host the competition and if I were them what would want to get out of it.  There was already a playground competition, so why release it again?  My thoughts were that 1. maybe Kaggle wanted to show the difference in quality of solutions for a free playground vs a prize value based solution competition, or 2. the sponsor wanted to get more out of the really challenging part of the problem, namely how to identify new_whale (N=0) and N=1 samples.  So my focus was on the latter and specifically how to identify as many N=1 samples as possible.

There are three main components to my pipeline:


 

 - **Keypoint matching** – old school approach with a few new school tricks
 - **Siamese network** – like many, [Martin’s previous work][1] formed the basis here
 -  **Post-processing** – to give low sample classes a fair shake

  

**Keypoint matching**
This accounted for &gt;80% of my final predictions, and was classic keypoint matching, one of the original low-shot methods.  I tried SIFT, ROOTSIFT, and a host of binary descriptors and matchers, there wasn’t a lot of difference between the different techniques.

The dataset here was in the sweet spot where brute force keypoint matching came into play.  7960 test images vs 15,697 train images is within the realm of reason.  I chose the pure brute-force method at full image resolution, all test images vs all train images, no bag-of-words or knn clustering of the keypoints.  There were a couple big challenges I had to overcome:

1. *Speed*.  Keypoint descriptors/matching can take up to 1-2s per image depending on your HW setup, but I used several tricks like indexing all keypoints to a hdf5 file, storing all keypoints into RAM during matching, and use of the great [faiss library][2].  Across two systems I could finish a brute force run of the full dataset in ~12 hours.
2. *False positives*.  The main issue with kp matching on this dataset was the false positives which had two sources: the background ocean and many of the bright points on the whale flukes.  I addressed this by using a unet to segment only the whale tail, and a custom xgboost model of the homography matrix to classify the final homography between image pairs as valid or not.

Final kp matching pipeline:
 - Extract all kps from train and test (raw images, full scale) into hdf5 files, restricting keypoints to unet predicted mask area of whale fluke.  Extracting from CLAHE preprocessed images worked best.
 - Matching:
	a. Loop through all test/train pairs
	b. Match keypoints using faiss
	c. Double homography filtering of keypoints (LMEDS followed by RANSAC)
	c. xgboost prediction to validate homography matrix
	d. if # of matches &gt; threshold, then use prediction



**Siamese network**
This is the weakest part of my pipeline, there are other posts indicating much stronger networks than what I used.  I just adapted Martin’s code a bit, and used an ensemble of InceptionResNetV2, InceptionV3, and ResNet50.  I didn’t add in any augmentations and stuck with grayscale images, nothing fancy.
To help training move on a little quicker, I did a fair amount of pretraining of the backbone network before feeding it in the Siamese network, which seemed to help.  My pretraining pipeline was:
 - train classification on top 200 classes
 - fine-tune on all classes where N&gt;8 (~576 classes)
 - fine-tune on all classes
 - fine-tune on all classes + mixup + image size 384x384 


**Post processing**
I found some similarities in the data between this competition and the Statoil Iceberg challenge, so I was able to use some of the same tricks from Weimin and my [winning solution][3] there, mainly that there were insights from test predictions that could be used to further enhance the test predictions. 


When analyzing the resulting prediction matrix from the Siamese network, I noticed that there was always a handful of the same train images that disproportionately dominated the top-5 positions.  This got me thinking that I needed to find a way to either suppress the dominate predictions or figure out how to get the N=1 classes a more fair chance to rise to the top of the prediction pool.


The idea I came up was pretty simple: instead of looking at the prediction matrix in the traditional way of “which train image is closest to my test image”, I transposed the matrix to now look at “which test image is closest to my train images”.  When I limited the transposed matrix to the N=1 samples, I found that I could use a new threshold along the train axis for the N=1 train samples.  This was highly effective at generating many more of the correct N=1 samples in my top-1 prediction.  I’m sure there are better ways of accomplishing the same goal.

I was surprised by the number of mislabels other competitors found, and thanks to [Alex Mokin and the contributors to this post][4] I took advantage of making sure the redundant classes were accounted for appropriately.

**Pipleline weaknesses:**
Again, thinking like the sponsor, they may not love my solution for a couple of reasons: 1. very computationally expensive, especially the keypoint matching pipeline, and 2. the difficulty to convert the pipleine into an easy way to do single image inference due to the post processing.

I would probably take someone else’s solution who has a strong siamese network and drop it into my pipeline as a pure replacement.  This would require retuning of the post processing pipeline but it’s possible.

**Pipleline strengths:**
I think the keypoint pipeline is pretty strong, without a lot of opportunity to squeeze more if using traditional keypoint algorithms.  The unet and xgb model incorporation into the pipeline really helps cut the false positives to be negligible.


  [1]: https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563
  [2]: https://github.com/facebookresearch/faiss
  [3]: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge
  [4]: https://www.kaggle.com/c/humpback-whale-identification/discussion/81885
