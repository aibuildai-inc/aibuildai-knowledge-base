# 6th place solution

Competition: feedback-prize-english-language-learning
Rank: #6
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369567

## Overview
I noticed that the correlation between 6 classes is quite high in my baseline models.  So I tried to relax it and created models for each class, which I would call  “separate models” here. I also trained single models (no-separate models) but with separate heads. In the end, my solution is a class-wise weighted average of 4 separate models and 5 single models, where I used the gp_minimise function from scikit-optimize to determine the weights.  
Note: a 5-fold separate model uses 5 x 6 = 30 models and it took around 1h for GPU submission.

selected submission: CV=0.4422, PublicLB=0.436362, PrivateLB=0.434121  
(best CV was my best PrivateLB)

| exp | type | model | fold | CV | Public LB | Private LB |
| --- | --- | --- | --- | --- | --- | --- |
| 14_v1_01 | separate | deberta-v3-base | 5 | 0.4524 | 0.4371x | 0.4380x |
| 14_v1_07 | separate | deberta-v3-base | 10 | 0.4512 | 0.4374x | 0.4370x |
| 14_v1_10 | separate | deberta-v3-large | 5 | 0.4518 | 0.4411x | 0.4370x |
| 14_v1_12 | separate | deberta-large | 5 | 0.4534 | 0.4409x | 0.4388x |
| 29_v1_02 | single | deberta-v3-base | 5 | 0.4557 | 0.4424x | 0.4416x |
| 29_v1_04 | single | deberta-v3-large | 5 | 0.4531 | - | - |
| 29_v1_11 | single | deberta-v3-large-squad2 | 5 | 0.4526 | - | - |
| 29_v1_14 | single | deberta-v2-xlarge | 5 | 0.4569 | - | - |
| 29_v1_15 | single | deberta-xlarge | 5 | 0.4552 | - | - |




training code: https://github.com/tikutikutiku/kaggle-feedback-prize-english-language-learning
inference code: https://www.kaggle.com/code/tikutiku/feedback3-inference2

## What Worked 
[Worked] separate model 
- What you did:  created models for each class
- Result: CV+0.003x, Public LB+0.005x, Private LB+0.003x

## What Didn’t Work 
 [Didn’t Work] awp 
 [Didn’t Work] pre-training with fb1 data 
 [Didn’t Work] detector pre-training with fb1 data
 [Didn’t Work] pseudo-label with fb1 data : models trained with pseudo-labels did not contribute to ensemble CV  
 [Didn’t Work] random masking augmentation 
 [Didn’t Work] lgb/catboost stacking

## Additional Context : 
Since the separate model was too slow for submission, I gave up the CPU track . 

## Thanks and Acknowledgements: 
Thanks to kaggle and the organizers for having this 3rd feedback competition! I enjoyed all of the three feedback competitions and had a great time with the team and solo play :) 

## Team Members: 
@tikutiku
