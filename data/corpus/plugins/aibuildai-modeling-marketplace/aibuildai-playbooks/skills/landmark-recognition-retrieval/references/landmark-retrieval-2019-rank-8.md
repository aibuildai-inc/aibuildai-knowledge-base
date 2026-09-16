# 8th Place Solution: Clova Vision, NAVER/LINE Corp.

Competition: landmark-retrieval-2019
Rank: #8
Source: https://www.kaggle.com/c/landmark-retrieval-2019/discussion/94540#latest-545876

# Approach by Clova Vision, NAVER/LINE Corp.
Thanks to the organizers for the awesome challenges (both recognition and retrieval), and congratulations to all participants!    
We briefly explain our approach here, and hope the future participants can benefit from this :)

----------------

## Quick summary of pipeline
+ [step 0.] Dataset Clustering    
+ [step 1.] Learning Representation
+ [step 2.] Feature Ensemble
+ [step 3.] DBA/QE + PCA/Whitening
+ [step 4.] DELF / Diffusion   

-----------------------

### (1) Dataset Clustering    
At the very beginning of the challenge, stage1 train/test/index datasets were all released, and the labels were provided for the train set only. We wanted to have the model biased to the test/index data distribution since it might help to achieve a high score in the challenge. To do so, we did clustering on stage1 test/index datasets and filtered out noisy clusters using several rules. The virtual class ids were given to each cluster and added to the train v1 dataset to use it for training. In this way, we got approximately 0.01 mAP increase on stage1.

At the stage2, we found the newly released train/test/index dataset is super noisy comparing to the stage1. We also made clusters on train/test/index datasets to remove the "garbage images" based on the constructed clusters.      

Finally we train our model using the datasets below:    
+ TR1: train v1
+ TR2: train v1 + test/index cluster v1
+ TR3: train v1 + test/index cluster v2 (noise cluster)
+ TR4: train v1 + test/index cluster v1 + train v2 (garbage removed) + test/index cluster v2 (garbage removed),      

where v1 is from stage1, v2 is from stage2.   


### (2) Learning Representation
We've tried to train various models by combining different backbones, losses, aggregation methods and train sets,

+ training data: TR1, TR2, TR3, TR4
+ backbones: resnet50, resnet101, seresneet50, seresnet101, seresnext50, seresnext101, efficientnet
+ losses: xent+triplet, npair+angular
+ aggregation methods: GeM, SPoC, MAC,     

which results in  4 x 7 x 2 x 3 = 168 combinations (or more...)      
In general, GeM worked well. We didn't have much time to fully fine-tune the efficientnet (tensorflow released version) on the google-landmark-challenge 2019 datasets so we early stopped the training.     

### (3) Feature Ensemble     
Based on the trained model, we extracted the features and found the best combinations for the ensemble. To test the result on a small dataset, we sampled validation set (VAL_V2) which is a subset of stage2 datasets. We run auto-search process which automatically concatenates 4 different off-the-shelf features which are randomly chosen every time, and evaluates the performance of the concatenated feature on VAL_V2. We found the mAP result on VAL_V2 is aligned with the result on real submissions, and this trick helped us saving submission trial and time.      

Finally, we found the best combinations:    

+ seresnet50 / SPoC / TR4 / npair+angular
+ seresnext50 / SPoC / TR4 / npair+angular
+ resnet101 / GeM / TR1 / xent+triplet
+ seresnext50 / GeM / TR3 / xent+triplet

The features were L2-normalized before the concatenation.

### (4) DBA/QE + PCA/Whitening
DBA/QE is a well-known technique for boosting the performance. We found 10-nearest neighbors for each data points for this. Based on our model, we got 9~10% mAP increase at stage1. The PCA/Whitening improved mAP a bit more when the output dimension was set to 1024.     

### (5) DELF / Diffusion

We applied DELF in two ways:    
+ reranking top-100
+ Diffusion with SV

we found simply reranking the top-100 retrieved result was not really helpful. We tested only one case, but it showed +0.00018 performance increase after DELF reranking (it might depend on models though).    
Instead, we used DELF matching score for graph construction for diffusion, and we got the better performance increase in this case. The gain was about 0.002 in mAP.   

For the last step, we applied diffusion on the features after DBA/QE + PCA/Whitening. It always improved the performance, but we performed a limited number of experiments because the diffusion process was very slow.

-----------------------

## Our Thoughts on Frequently Asked Questions   

__Q1. Our Best Single Model?__           
The stage2 mAP of best single model is approximately in range of [0.1718 ~ 0.1995] in public, [0.1878 ~ 0.2211] in private with 1024-dimensional feature. The high score model in public did not necessarily achieve a high score in private.   

__Q2. Local Descriptor vs Global Descriptor?__     
It's not a good idea to use DELF directly to retrieve images from 1M index images due to excessive computation. Suppose we have top-K ranked candidates retrieved by global descriptors, we can rerank the order using local descriptors with geometric verification.
In our case, the gain was very slight though.    

__Q3. Things that didn't work?__        
+ Ensemble with features extracted from multi-resolution input images. (didn't work)
+ Explore-Exploit Graph Traversal (EGT) (slow on our implementation)   
+ Detect-to-Retrieve (D2R) (cropped regions from D2R landmark detector were used for feature extraction, and the result was worse.)


** for more detailed experimental results, please refer to the paper we upload to arxiv. (https://arxiv.org/abs/1907.11854)
