# 6th simple solution, pre-training, single model private 0.66927

Competition: understanding_cloud_organization
Rank: #6
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118017

First of all, I would like to thank the hosting organization that hosted this competition and Kaggle. Like any competition, this competition was also hot until the end.  So, I want to congratulate Kagglers who struggled until the end of this competition.

I will summarize and write down the part of my solution that you will be interested in. It's `pre-training`

# pre-training
The challenge of this competition is to segment according to the shape of the cloud.  Therefore, I tried to pre-train the model to learn the shape of the cloud.



Because the clouds are white, I generated `cloud_mask` with the threshold of "pixel &gt; 115". Then, I used it as a label. (Since the total number of image files is 9244, the cloud_mask also generates 9244.)

After pre-training, I tried a 2nd-stage training.
Pre-trained(1st stage-training) model are used as the initial value of 2nd-stage model weights.



This training process boosted my CV 0.005~0.01. So, my single model score is as follows.

| model | private | public |
| --- | --- | --- |
| efficientnet-b4, unet | 0.66927 | 0.67437 |
| efficientnet-b4, fpn | 0.66827 | 0.67508 |



The rest is not special, so I'll skip the description. 😁 
Thanks for your reading!
