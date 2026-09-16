# 29th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #29
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369559

First of all, I would like to thank Feedback competition host and all the participants.
I was able to test much of my knowledge from past competitions in this competition.
My solution is shown below. I have reflected that I could have made more detailed tuning to each solution method, and I hope to make the most of this in the next competition.
# Overview
1. I had **six pre-trained models** train with the given train data. **AWP** was used in my all training.
**six pre-trained models**: deberta-v3-large, deberta-v3-base, deberta-v2-xlarge, deberta-large, bigbird-roberta-large, bigbird-roberta-base
2. Six trained models were used to **Pseudo-Label** the FB1 essay texts.
3. Pseudo Labels were added to the given train data and again had six pre-trained models train.
4. The predictions of the six models plus SVR were ensembled by weighted average.
# What Worked
- **AWP**
 - I have tried AWP several times in past competitions, but all failed. And for the first time, I have succeeded in improving my score with AWP.
 - The parameters worked well with adv_lr=2e-4, adv_eps=1e-3, referring to the discussion.
cf. https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/349594#1932041
 - CV improve (single model): 0.0007
- **Pseudo Labeling**
 - No leaks. I knew empirically from past competitions that if I am not careful with folds, they will leak.
 - CV improve (single model): 0.0003 ~ 0.0026
- **Diversity by training pattern**
 - I was able to create diversity by dividing the training into the following 1 and 2 training patterns. These ensembles improved the score.
1: train one epoch with Pseudo Labels, then train several epochs with given train data
2: train several epochs with given train data and Pseudo Labels at the same time
- **Weighted average Ensemble**

 | Model | Folds | CV | Public LB | Private LB
 | --- | --- | --- | --- | ---
 | deberta-v3-large | 5 | 0.4473~0.4495 |  | 
 | deberta-v3-base | 5 | 0.4512~0.4522 |  | 
 | deberta-v2-xlarge | 5 | 0.4475~0.4522 |  | 
 | deberta-large | 5 | 0.4510~0.4538 |  | 
 | bigbird-roberta-large | 5 | 0.4577 |  | 
 | bigbird-roberta-base | 5 | 0.4643 |  | 
 | svr | 5 | 0.4526 |  | 
 | Ensemble |  | 0.4435 | 0.437183 | 0.435225
# What Didn’t Work
- **Down sampling of Pseudo Labels**
 - I considered some of the Pseudo Labels to be unreliable and removed those with large std (>0.44/3) of predictions for each model.
However, the CV did not improve.
 - CV (Weighted average Ensemble): 0.4435 -> 0.4437
- **Efficiency Prize Measures**
 - For Efficiency Prize, I had deberta-v3-small train with all given train data (1 fold) and all Pseudo Labels.
 - The inference time for the test data was 40 minutes, which was too long.
I should have set `batch_size=1` and tried deberta-v3-xsmall. (Thanks to @tmhrkt for the advice)
 - **Private Score: 0.438587** (Not bad!)

This competition was a lot of fun, with the Efficiency Prize as before, and a commemorative gift for the top 50. Thank you so much, hosts!
I have already started warming up for FB4😉
