# 5th place solution

Competition: feedback-prize-english-language-learning
Rank: #5
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369578

Thanks to Kaggle and hosts for hosting the 3rd addition of the Feedback competition.

# Overview 

My solution is based on an ensemble of multiple finetuned NLP transformer models. Additionally, I employ two rounds of pseudo tagging on old Feedback data.

I follow a lot of our advice from the second Feedback competition described [here](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536) and worked for around two weeks on this competition. I split my time working around 50% on accuracy, and 50% on efficiency solution. 

## Cross validation

In general, I observed very good correlation between local CV and public LB. As the data is very small and the metric is RMSE, the local scores can be quite shaky. To that end, for each experiment I was running, I trained three unique seeds and always only compared the average of these three seeds. So for example, if I would want to compare LR=1e-5 vs. LR=2e-5 I would run for each of those two experiments three separate seeds for a single fold, and only if the average of the three seeds improves, I would run on all my 5-folds, and then again compare 3-seed blends to make sure.

This allows to bring more trust to my experiments and as the data is really small, this was in general possible for me to do.

## Modeling

The problem at hand is very much straight forward, feed in the text to a transformer model, apply some pooling, add a linear head, and predict regression targets. I used combinations of the following variations of the training routine for my final ensemble:

Token length:
- 512
- 1024
- 2048

All my models are trained and predicted with dynamic padding.

Pooling:
- CLS Token
- GeM Pooling

Backbones:
- Deberta-V3-Base
- Deberta-V3-Large
- Deberta-V2-XL
- Deberta-V2-XXL
- Longformer Large
- Roberta-Large

I usually run 3 epochs for most of my models, all with cosine decay learning rate always picking the last epoch. I use differential learning rate for backbone and head of the model. I do not use any other techniques suggested in forums like differential lr across layers of backbone or reinitialization.

As always, I retrained my models on full data for final subs, but also blended some fold models in as I had lots of runtime left.

## Pseudo labels

I followed our routine from 2nd FB competition and employed two stages of pseudo labeling following these steps:

1. Train an ensemble of models only on the given train data
1. Run predictions on the previous Feedback competition data excluding this competition's data
3. Use pseudo labels from this extra dataset and apply it to modeling by pre-train models on the pseudo labels and finetune it only on the given train data afterwards. 
4. Repeat steps 1-3 three times using an ensemble of models trained on pseudo labels now

By doing pre-training and finetuning, I did not need to adjust the distribution of the pseudo labels, because I am doing the final adjustment on this competition's data. This allowed me to use all prior data without issues.

## Ensembling

For most of my subs I just did usual average across seeds and models. My final best sub is a Nelder-Mead optimized ensemble of models, where I optimize the ensemble weights separately per target column. To not overfit too much on CV, I added weight bounds between 1 and 3 on the weights.

Actually, I could have trusted the local optimization even more, I have an unselected sub from couple of days ago with best local CV that would score #2 on private, which has unrestricted weights for the ensemble, also with negative weights, but it felt a bit too risky and as I only did one sub for best CV, I chose a bit more of a conservative one.

## What did not work

- Augmentations (as always, specifically tricky with regression)
- Different losses
- TFIDF
- Other backbones (such as T5, GPT, etc) - Deberta is so strong
- 2nd stage models / stacker models

## Final subs

I did one sub based on my conservative best CV score, which also was my best selected private LB sub (although I had a better one unselected). I did one sub based on best public LB which was clearly worse on private LB. And I spent one sub on efficiency.
