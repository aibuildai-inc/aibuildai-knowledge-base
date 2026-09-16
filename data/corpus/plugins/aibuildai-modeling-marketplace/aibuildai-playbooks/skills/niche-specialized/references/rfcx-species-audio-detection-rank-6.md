# #6 Solution 🐼Tropic Thunder🐼 (journey and ensembling schemes)

Competition: rfcx-species-audio-detection
Rank: #6
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220981

Thanks to Kaggle and the Host for this competition. It's going to be special for me, I have got here my first Gold medal and my upgrade to Master tier in Competitions.

Thanks a lot to my teammates: @jpison @pavelgonchar @antorsae. We worked together as a real team. 

After @antorsae post (https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220446) with his perspective, I'd like to give more information that could be useful to understand the influence of the differents experiments we did with our **Tropic Model**.

First, find here **our journey, under submissions perspective**:
[Submissions]
@jpison and @amezet join in a team on January 19th
@pavelgonchar joined the team on February 1st
@antorsae joined the team on February 10th

We worked in two different models (**without hand labelling**):
1. **SED model**. We worked with this base model, improving with mixup, intelligent cropping, better schedule: https://www.kaggle.com/gopidurgaprasad/rfcx-sed-model-stater.
We'd like to thank @gopidurgaprasad
We reached with this model 0.91760 in Public (0.92270 in Private)
2. **Tropic Model**
@antorsae explained in his post the details of the models.

We learnt that the more diversity we have, the better score we got. So we did the experiments changing a lot the barebones, taking different from the complete list of TIMM models:
https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv

To select our two submissions, we prepare two scheme of ensembling:

**Submission selected 1:**
[Scheme 1]

**Submission selected 2:**
[Scheme 2]

We could did experiments very fast because we have a lot of computing power. For this competition we used the following GPUs:
@antorsae
6x3090
@amezet
3x3090 + 2x2080Ti
@jpison 
1x3090 
@pavelgonchar
2x3090
