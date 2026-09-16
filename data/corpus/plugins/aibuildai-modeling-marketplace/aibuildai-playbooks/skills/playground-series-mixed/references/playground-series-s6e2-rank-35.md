# # 35 Place - Time for a write up

Competition: playground-series-s6e2
Rank: #35
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/35-place-time-for-a-write-up

I am really happy as I share this solution writeup (my first writeup) since this has been my best playground series competition till now with a rank of #35 and if we see the scores, it's a #3 ranking score.

Before I start, thanks to the whole Kaggle community with so many nice people around, most of what I have learned, I learned from here.

# Solution (the final ensemble) - Last things First

All my models were really very simple.

| Model | CV | Private LB | Weight |
| --- | --- | --- | --- |
| TABM | 0.95562 | 0.95527 | 0.29 |
| XGBoost with Target Encoding | 0.95569 | 0.95530 | 0.42 |
| CatBoost with Target Encoding | 0.95566 | NA | 0.14 |
| CatBoost with Base Features as Categorical | 0.95561 | NA | 0.14 |
| CatBoost with Base Features + Base Features as Categorical | 0.95567 | NA | 0.30|
| **FINAL STANDINGS** | **0.95574** | **0.95533** | |

I used hill climbing for ensembling, that was something new for me. I had a total of 17 models, but rest all got 0 weight in the ensemble, that's why they are not listed here.

# What and How?

I started participating in this episode from day 1. 

**[Logistic Regression](https://www.kaggle.com/code/rattans/logistic-regression-ps-s6e2)** - I started with the simplest linear model for classification task to find the probabilities of Heart Disease.

My best Logistic Regression would have ranked under 1000 which is a great thing, for beginners specially since they are just starting, instead of running to GBMs or NNs, linear models are a good starting point.

**XGBOOST** - Moving further I used XGBoost. To be precise, I created 7 XGBoost Models. In 6 of them, I tried different feature engineering methods like one-hot-encoding and target-encoding. Although, there were a lot of feature engineering ideas that I had found through past discussions were left to be explored. Maybe that would have created a better difference. One of my XGBoost model was a stacking over LogisticRegression.

**TABM** - Later in the competition, there were discussion over models like TABM and RealMLP, I used TABM for the first time and it was effective.

**CatBoost(last day)** - I was not much active for the second half of the competition but I had to try CatBoost, so I forked all my XGB models and replaced them to CatBoost on the last day, this is the reason that stats around Public and Private LB of those models were not present above since I refrained from submitting them as I had only 10 subs left and unless I found a great CV submitting would be of not much use. Instead, I saved the oof and submission files for ensemble.

**Sequential NN** - I even used sequential NNs from keras but they could not even outperform Logistic Regression. I mean maybe they would have, but I could not make them do so.

**Ensemble** - On the final day with 17 models trained, it was time for a final ensemble. I used Hill Climbing, and I really forgot from where I copied that code. But thanks to whoever it was, and sorry I could not remember and mention you.

# Learning and Experiences

**Playground Series are meant for the purpose of learning, so it's an advice, request or suggestion, feel free to call whatever you want to call it, "we should use them to experiment and learn as much as we can".**

My key learnings:

* Used TABM for the first time, came across REALMLP too but don't know where that model disappeared from my notebooks (it gets a bit messy handling so many notebooks on kaggle at a single time)
* Used hill climbing to ensemble for the first time.

Here are some of my Experiences:

* Playground Series = Playground to Experiment
    * Experiment with different models
    * Experiment with different feature engineering ideas
    * And experiment is something I should not tell what to experiment with, try anything and it's an experiment for you. 

* Blind Blending = A drunk person with a car [Harms himself, harms others]
    * I don't think I need to talk much about this, there are already many discussions around it.
    * Just two sentences - If you are here to learn, then learn. If you are not here to learn, then maybe you are at some wrong place.

* Chase LB = You will fall high
    * Focus on CV - either you will rise or you'll stay stable.

# Conclusion

I kept everything really very simple. New features I added were mainly one-hot-encoding of features and target-encoding. There was still a lot of room to experiment and explore. Whatever it may be, this was really very happy ending for me.
