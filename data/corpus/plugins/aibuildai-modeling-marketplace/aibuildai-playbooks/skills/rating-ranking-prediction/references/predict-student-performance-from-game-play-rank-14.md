# 14th Place Solution Joseph Part

Competition: predict-student-performance-from-game-play
Rank: #14
Source: https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420041

Thank my teammates for their efforts, I have learned a lot from them. Luckily we don't shake-down too much. Now I would like to introduce my solution to you. 
# Modeling
My modeling method is like a 'cumulative' one: using 0-4 part data to generate the train set of q1-q3, using 0-4 and 5-12 part data to generate the train set of q4-q13, using 0-4, 5-12 and 13-22 part data to generate the train set of q14-q18. **Question is also a feature**. It has merit that I don't need to save 'historical' data. And this one is time-saving and costs about 40min for inference.

# Feature Engineering
There are my FE ideas:
- **basic agg** features: eclipse_time_diff sum, count and max of each group, each level, each event_name, ..., each text; eclipse_time_diff sum, count under a particular room_fqid and an event_name, etc.
- **behavior-change** features: the number of room change, and the number of room change under each level; the number of text_fqid change, and the number of text_fqid change under each level, etc.

In this picture, we can see a **room change** behavior, we calculate the change times and average to characterize one's ability to understand and reason. Some of them have pretty high feature importance.

- **Meta** features: Besides basic `groupby` feature engineering, I add the meta feature for 5-12 and 13-22 groups. There are two way to use them:
1. each question's `predict_proba` as a feature, 5-12's model includes features **q1_proba**, **q2_proba**, and **q3_proba**, 13-22's model includes features **q1_proba**, **q2_proba**, ... **q13_proba**.
2. mean of all question in one group as a feature,  for instance, 5-12's model includes a feature **mean_of_q1-q3_proba**, 13-22's model includes features **mean_of_q1-q3_proba**, **mean_of_q4-q13_proba**.

# Models
For my part, I use 9 models for my ensemble. They are 4 xgboost, 1 lightgbm, 2 dart, 2 catboost. The private-best single model is a dart, which achieved **Public 0.704** and **Private 0.704**. The public-best model is a xgboost, which achieved **Public 0.705** and **Private 0.698**. My dart notebook [Game-Play-LGBDart[INFER] Private LB 0.704](https://www.kaggle.com/code/takanashihumbert/game-play-lgbdart-infer/notebook)
# The difficulty
I think the most difficult part of this comp is to establish CV and choose the threshold and the submissions. As you can see, my dart model and xgboost model in the same CV strategy vary wildly. It's beyond my expectation. I even have no confidence to give my dart models a bigger weight. **I believe many teams didn't choose their best results.**

Finally, I would like to pay tribute to all kagglers who share their ideas. See you next game.
