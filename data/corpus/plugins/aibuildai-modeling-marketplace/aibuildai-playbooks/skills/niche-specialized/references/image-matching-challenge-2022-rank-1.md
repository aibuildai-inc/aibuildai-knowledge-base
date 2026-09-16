# 1st Place Solution

Competition: image-matching-challenge-2022
Rank: #1
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/329131

Thanks to all participants for their hard work in the competition.
We are honored to have achieved a great result of 1st place in this competition.

We would also like to express our deepest gratitude to the organizers for organizing such a wonderful competition.
Thank you very much.

Finally, thanks to my amazing teammates @gmhost,  @xiuqi0, and @forcewithme the universe.
I believe that their great insight and experimentation brought about honorable results. Thank you very much!

# OverView
Our solution is based on an ensemble of matching results from three different models: LoFTR, SuperGlue, and DKM.
They all use publicly available pre-trained weights and have not been trained or fine-tuned.

Significant score increases were also achieved by matching at multiple resolutions and by cropping only the covibility region before matching.

Our solution is a 2-stage method and the framework is as follows.
1. We first concat the keypoints produced by LoFTR with a 840 resolution and keypoints produced by Superpoint_SuperGlue with 840, 1024, 1280 resolution. This is the first stage of our framework.

2. We **use DBSCAN to get clusters** containing the top 80~90% matching keypoints. Generate a bbox for each image with those keypoints, and crop it. We call this method mkpt_crop. Mkpt_crop can efficiently filter outliers and extract key areas.

3. Use LoFTR, dkm, SuperPoint_SuperGlue to rematch on the key area. This is the second stage of our framework.

4. concat the keypoints produced by stage1 and stage2. Then do RANSAC and get fundamental matrix.

[[solution-framework.png]](https://postimg.cc/JDhJzSCY)

# KeyPoint
* mkpt crop
* ensemble multi-resolution matching

### mkpt crop (matching keypoints crop)
Based on the hypothesis that regions not shared by two images are unnecessary for matching, we developed the idea of **mkpt crop** (matching keypoints crop), which crops images efficiently and accurately.

The mkpt crop crops covisibility regions in the following way.
1. We use pre-cropped images (original images) to match each other 
2. Clustering the matching points of each image with DBSCAN
3. Extract clusters containing the top 80%~90% matching points to remove outlier regions
4. Crop the outer boundary of the cluster

[[mkpt-crop-image.png]](https://postimg.cc/zLwvY7Lt)

Since a well pre-trained model finds quite a lot of correct matching points, we were able to find covisibility regions with high accuracy by thinning out the outliers to some extent. 

The models used in Step 1 were LoFTR and SuperGlue. At first, only the LoFTR results were used to calculate the crop area, but by adding the SuperGlue results, it was possible to cover the covisibility areas that LoFTR could not find, thus increasing the score significantly.

In addition, mkpt crop is a very lean and efficient method because the matching points computed in step 1 can also be used for the final ensemble.

We also tried to crop by segmentation using Mask2Former, but since segmentation crop cannot determine whether or not a region is common to two images, there were inevitably many wasted regions, and mkpt crop gave better results.

However, mkpt_crop has the risk of removing small covisibility regions because it removes clusters with few matching points. Therefore, our team added the matching points from the original image to the ensemble to achieve more stable matching.

### ensemble multi-resolution matching
We used LoFTR, SuperGlue, and DKM to perform matching at various resolutions and ensemble the results.
The final model used was as follows.  
| #    | Model    | input image | resolution |
| :--- | :---     | :---        | :---       |
|1     | LoFTR    | original    | 840        |
|2     | SuperGlue | original    | 1280       |
|3     | SuperGlue | original    | 1024       |
|4     | SuperGlue | original    | 840        |
|5     | LoFTR    | mkpt crop   | 1280       |
|6     | DKM      | mkpt crop   | 840        |
|7     | SuperGlue | mkpt crop   | 1024       |
|8     | SuperGlue | mkpt crop   | 1280       |
|9     | SuperGlue | mkpt crop   | 1536       |

(#1~4 are used for mkpt_crop calculation)

The matching points output by these models are rescaled to fit the original image size and then concatenated using np.concatenate to compute the F matrix.

We believe that changing the input resolution led to an improvement in scores because the matching points obtained were different and more diverse.

# Other tips
### Use of pretrained models
All of our solutions this time use a publicly available pretrained model.

Since the quality of the training and test data differed in this competition, we did not conduct fine tuning using the provided training data because we thought it would be less effective.

### RANSAC parameter
We used USAC_MAGSAC to compute the F matrix.

However, the parameter settings in the public notebook took a long time to calculate and resulted in a timeout, so we repeated several experiments and adopted the following parameters, which are fast and do not change the accuracy much.  
`cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.2, 0.99999, 50000)`
With these parameters, the above model configuration took approximately 9 hours to complete the process, yielding a score of public:0.86287/private:0.86343.

Incidentally, although not selected in the final submission, a solution that excluded the #4 model (SuperGlue/original/840), increased the DBSCAN eps parameter slightly, and employed the following parameters yielded private:0.86516.
`cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.19, 0.99999, 50000)`
We have not experimented enough with the model configuration and the combination of RANSAC parameters, so there may still be room for improvement.

# Things are also useful but not as effective as our final solution
### TTA
Finally we did not use TTA.
We initially used horizontal flip TTA, but found that multi-resolution improved scores more, so we eliminated TTA and increased the number of models used in multi-resolution.

### Other Matching Methods
We also tried matchformer and keynet_affnet_hardnet, but ultimately did not adopt them.
* matchformer: Slow, can improve by about 0.002.
* keynet_affnet_hardnet: can improve by about 0.001

# Things do not work
1. Train LoFTR, Superpoint, and SuperGlue.
2. Larger resolution for dkm
3. Use Segmentation to filter moving object(e. g. cars, people)
4. Adding ASLFeat
5. Adding ALike
