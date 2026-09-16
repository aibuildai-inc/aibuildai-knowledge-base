# 3rd place solution sharing: A Deep Mixture Model with Online Distillation

Competition: youtube8m-2019
Rank: #3
Source: https://www.kaggle.com/c/youtube8m-2019/discussion/112929

At first, I would like to thanks google research for providing another interesting video understanding challenge. This competition really provides fun to many of my weekends in the last 4 months.  

Overall, my solution follows the widely-used system design: candidate generation and ranking. 

A quick offline analysis suggests that the top20 topics(among 1000 topics) cover over 97% of the positive labels. The segment level classifier is directly finetuned from video-level classifer(with the same structure).

I found larger model can generally perform better in the video dataset but will quickly overfit the smaller segment dataset. In this competition, I tried another approach to increase model capacity by training multiple models. Our final model is a 2-layer mixture model with online distillation.  Each of the MixNeXtVLAD model is a mixture of 3 NeXtVLAD model. So in total, we trained 12 NeXtVLAD models in parallel using 4 Nvidia 1080 TI GPUs.  The online distillation part can effectively prevent the whole model to overfit the smaller dataset. 



 



More details about the model will be included in the research paper and shared in this post once I finish the writing : )

If you are interested in the performance of models I have tried, following are the results:


 Iuse all the available data for training, including the validation set, because the performance on local validation dataset is highly aligned with public LB.
