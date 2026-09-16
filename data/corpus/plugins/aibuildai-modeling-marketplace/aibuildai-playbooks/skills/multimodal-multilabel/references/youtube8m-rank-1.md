# 1st Place approach (WILLOW)

Competition: youtube8m
Rank: #1
Source: https://www.kaggle.com/c/youtube8m/discussion/35063

Hi,
Sorry for the late post, we wanted the paper to be nice to read :).

Here is the report: https://arxiv.org/abs/1706.06905

Code:
A github repo to reproduce a winning submission (not the best one). It should do 84.698 on the private. It is a very simple average of 7 models checkpoints.
https://github.com/antoine77340/Youtube-8M-WILLOW

And here a github repo containing our Tensorflow toolbox (LOUPE), implementing all the learnable pooling modules with their Gated versions (see the paper for more information):
https://github.com/antoine77340/LOUPE

The paper mainly focus on the contributions. It might lack some details concerning the kaggle best submission (which I think is not very important as a simple ensemble of 7 models already achieve the first place). If you really need clarifications, I will be happy to answer your questions.

Antoine for the WILLOW team
