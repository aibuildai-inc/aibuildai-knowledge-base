# 9th Place Solution

Competition: predict-student-performance-from-game-play
Rank: #9
Source: https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420046

First of all, I would first like to thank all the participants who dedicated so much time and effort to this competition, as well as the hosts and the management. While I still believe there are areas in which the administration of the competition could be improved, in this post, I will focus solely on discussing my solution.


## Overview
I didn't do anything particularly special.

Mainly, I just kept adding features to improve the accuracy of the single model. For each question, I built models using LightGBM and Catboost, and took the simple average of the two models. I used the models with the highest CV scores as the final candidates. The high CV submit also got almost the best score for private.

- LightGBM CV：0.7018 LB(Public)：0.703 LB(Private)：0.702
- Catboost CV:0.7011 LB(Public)：0.7 LB(Private)：0.701
- Merge(final submit) CV:0.7024 LB(Public)：0.7 LB(Private)：0.702


## Features
Rather than introducing each of the numerous features I created, I'll share my overall approach and discuss a few specific features that particularly contributed to the accuracy.

As already demonstrated in public notebooks, an important element in this competition was "how much time one spends playing." To delve deeper, I felt that "how much time it took from a certain point to another" was crucial, so I created many features related to this.

- Checkpoint feature (as I named it)
In this game, there are events that almost every player will inevitably experience. For instance, every user will find a notebook and see the message "found it!" I identified these "events that almost every user goes through," and used the time taken between these events (i.e., the elapsed time from event A to event B) and the number of clicks as features. This seemed to significantly contribute to the accuracy.

I created such features in various patterns, like the elapsed time from viewing text A to text B, the elapsed time from one fqid to the next, the elapsed time from one room to the next, and so on.

- Other than this, I obviously included features like the time elapsed for each level and the average coordinates, as introduced in the public notebooks.

The number of features increases with the level group. Ultimately, the feature counts were as follows:
Level group 0-4: 3009 features Level group 5-12: 9747 features Level group 13-22: 18610 features


## Modeling Approach
- I chose to use 10-fold rather than 5-fold as it gave slightly higher CV scores (around +0.0005).
- I used straified Kfold with the number of correct answers for 18 questions of the user. So, the model is made with the distribution of the total number of correct answers of the users almost aligned. (However, I don't think it would be much different with a simple K-fold)
- For feature selection, I simply used the top 500 features based on their importance. To prevent leakage, I selected the feature importance for each fold, and retrained the model for each fold. For example, when training fold1, I first train with all features, then select the features using the fold1 model, and retrain the fold1 model with the top 500 features.
- I used the prediction probabilities of previous questions as features. For example, when predicting question 3, I used the prediction probabilities for questions 1 and 2. When predicting question 15, I used the prediction probabilities for questions 1 through 14, etc. (this improved CV by around +0.001)

- Inference took the following amounts of time:
LightGBM: 90min Catboost: 120min Merge: 150min
LightGBM's inference became significantly faster by compiling the model with a library called "lleaves."
https://github.com/siboehm/lleaves
I'm sharing my inference code. Features not mentioned here can be somewhat understood by looking at it.
https://www.kaggle.com/code/mhyodo/restart-model-merge-v1


## What Didn't Work
- NN models (I tried several types, such as LSTM and MLP, but none contributed to the CV)



Lastly, I've seen posts from others where the LB score was higher than the CV score, but in my case, they were pretty much the same. I was expecting some sort of shakeup, but I didn't anticipate making it into the top 10. I'm curious as to how those with higher LB scores achieved this, as I was unable to significantly increase my LB score. 
Anyway, thank you! If the ranking is confirmed, I can become a new GrandMaster!
