# 26th place solution [lucky shake up 1117th → 26th]

Competition: feedback-prize-english-language-learning
Rank: #23
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369525

Thanks to all the participants and hosts
I was lucky to win a silver medal with big shake up.

# Overview
finally cv is 0.44582 (public/private 0.439025/0.435032) (deberta-v3-large (x2) ensemble)
MLM on FB2 data (mask prob = 0.4)
CV improved slightly but LB worsened. 
However, adding it to the ensemble seems to have boosted the score.

single model cv
- 0.4490 (w/o MLM) (public/private 0.439145/0.436422)
- 0.4488 (w MLM) (public/private 0.442191/0.437079)

In my case, LB did not correlate well with the CV improvement.
LB did not change much when I changed from a single model to an ensemble model
(This doesn't seem to be the case with the top solutions, I don't know why)
I stopped working on the competition because LB did not improve. 
However, TrustCV seems to have been right.

## What Worked
- [High Impact] Idea
    - AWP
    - attention pooling (target column-wise)
- [Medium Impact] Idea
    - layer wise learning decay

## What Didn’t Work
- smooth-l1
- mask augmentaion
- lstm(gru) head

Small Tips
I experimented with different seed. And I worked to make sure that the cv was improved.
(This scheme is described in FB2 1st place solution)

Looking forward to fb4. Thank you.
