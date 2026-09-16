# 1st place solution summary

Competition: landmark-retrieval-2020
Rank: #1
Source: https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176037

[Update] solution arxiv link : https://arxiv.org/abs/2009.05132
~~[Update] submission to arxiv is on hold. Please refer to attached pdf paper.(modified version)~~
# 
 Great thanks to google and kaggle team for hosting this competition, and congrats to all participants who finished successfully. I really learned a lot during the competition through reading articles, analysing codes, and doing experiments.

I'd like to share my solution, and detailed solution will be uploaded to arxiv in a few days.

Model structure is as below.


# Basic Configuration
Validation set : 1 sample per class which has >=4 samples in GLD v2 clean dataset(72322/81313 classes)
Cosine softmax : s=determined by adacos, m=0
Weighted cross entropy : proportional to 1/log(class cnt)
Augmentation : left-right flip
Optimizer : SGD(1e-3, momentum=0.9, decay=1e-5)
Embedding Dimension : 512 for every model
Hardware : Colab TPUs

# Training Strategy
1.Use GLD v2 clean dataset to train model to classify 81313 classes
efn7 512x512 priv.LB:0.30264, pub.LB:0.33907
2.Take efficientnet backbone from step 1, use GLD v2 total dataset to train model to classify 203094 classes
 efn7 512x512 priv.LB:0.33749, pub.LB:0.36576

3.Take whole model from step 2, give increasingly bigger images to the model
 efn7 640x640 priv.LB:0.35389, pub.LB:0.39121
 efn7 736x736 priv.LB:0.36364, pub.LB:0.40174
4.Take whole model from step 3, set twice loss weight for GLD v2 clean samples
 efn7 640x640 priv.LB:0.35932, pub.LB:0.39881
 efn7 736x736 priv.LB:0.36569, pub.LB:0.40215

# Ensemble
1.736x736 efn7+efn6+efn5+efn5 weighted concat
(all train step3, weight : efn7=1.0, efn6=0.8, efn5=0.5)
priv.LB:0.38366, pub.LB: 0.41986
2.Same Config, with train step 4 for efn7
priv.LB:0.38677, pub.LB: 0.42328

# 
If you have any questions, feel free to ask.
Thank you.
