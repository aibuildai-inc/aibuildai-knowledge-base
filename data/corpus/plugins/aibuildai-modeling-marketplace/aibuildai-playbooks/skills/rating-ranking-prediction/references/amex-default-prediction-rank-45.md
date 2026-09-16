# 45th place with XGBoost in first Kaggle competition

Competition: amex-default-prediction
Rank: #45
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/347966

Some luck and some innovation lead to a surprising top 1% in my first competition!

49th overall, 5th out of solo rookies. 3rd out of amateur solo rookies. (7th and 10th place overall were soloing their first kaggle competition, but profile shows they are professional data scientists.) To say the least, I'm very happy with my result! Who knows, but I might've had one of the top 10 or top 20 single models with my 0.80798 XGB score on private.

I'm a professional software engineer at Intel, but only a ML hobbyist (started with stock market related some years ago), so professional problem solver, amateur at ML. I spent far too much time on this competition, but a lot of it struggling with basics like "Oh, that's not a pandas df, it's a cudf df. Now I can look at the right documentation. ...Oh, the kaggle version of rapids cudf (20.x) doesn't have this function call that I'm staring at the cudf v21.x documentation of right this second." Only more painful and recurring small issues even than it sounds. I didn't touch NN. I touched LGBM along with XGB, but even spending time on two models was a bit more than I could easily do.

I'll cover my submission report first, and hopefully later add my general takeaways from my first experience with kaggle competitions. I like semi-stream of consciousness long-winded posts, so buckle up! :)

# Submission Report
## Score and Result
49th place. 0.80798 private LB. 0.79889 public.

Went from 406th (ensemble of two of mine with LGBM dart public) -> 49th (my standalone model submission).

## Solution Overview
My solution was pure XGBoost model for each step. At the high level, ignoring the timeline of my journey of discovery, I did these things:
* Note: I used auc score not logloss nor amex metric for any optimization or tuning along the way
* Created the XGB Pyramid via pathfinding and pure theorycrafting.
  * https://www.kaggle.com/code/roberthatch/pyramid-api-for-easy-deployment
* Basic popular feature aggregation and a few of my own. Most notably moving averages, though I only used them on all statements. Hull moving average, and exponential averages. 16 total aggregations per base numerical feature.
  * https://www.kaggle.com/code/roberthatch/amex-feature-engg-gpu-or-cpu-process-in-chunks 
  * https://www.kaggle.com/code/roberthatch/exponential-averages-amex-feature-engineering 
  * I dropped B_29 and never looked back!
* Meta-feature: Predict for every row in train and every row in test, the chance of missing the next month's payment, meaning days overdue increase by a large value, and ending at or over 28.
  * https://www.kaggle.com/code/roberthatch/amex-fe-02-days-overdue-label 
  * Inspired by: https://www.kaggle.com/code/raddar/deanonymized-days-overdue-feat-amex 
  * Not just using that row's data to predict, I wanted to use that row AND all past data via backwards aggregation. To fit in memory, forward feature selection (on normal target predictions) using various shortcuts, ending with 280 features.
  * I also predict the chance that next month's days overdue will be non-zero as an independent meta-feature. 
  * Low on GPU and time ~48 hours to go, just do single five fold oof predictions. Better to average 3-5 models. Could predict last statement with all models, since they couldn't be used for training, but for simplicity only predict "oof" last.
* Main model was step two, taking the aggregated meta-features (2*16) and the 3000 other features, convert to float 16 for memory, and run the XGBoost Pyramid.
  * Train 4 models on entire train dataset with no CV using a set number of rounds based on inspecting when early stopping happened on CV models.

## Notes
I had the hardest time doing (and failing at) permutation importance. My biggest bottleneck was my own time, and I didn't want to write it from scratch, but I couldn't use sklearn with my xgb model, and the things I tried kept failing for various reasons (including attempting to do an sklearn version of my xgb model, and including trying to leverage a non-sklearn version of permutation importance library). And I think I gave up on the non-sklearn one just because it was so slow.

In any case, thats why I ended up doing homebrew forward feature selection. Which I spent way WAY too much time on (but was kinda fun). At first I did selection only from last. Then added max, then e7. (from other experiments I had done forward feature selection of entire aggregation styles, and if forced to do only 3x aggregations, last, max, e7 was best for me. I got 70 features individually before finally stopping that slow approach. That allowed me to split the base columns into three groups, the "good" the "decent" and the "didn't seem good". So I did grouped forward feature selection based on the 16 aggregations, done on 1 of the three base feature subgroups, so 48 groups in total of 40-80 features each. I got to 3xx, but ran out of memory and to keep moving forward backed up to 280 features total. Not all base features were represented at all.

I had a ton of other random ideas, but all along was convinced that predicting using the test sets and days overdue was my best single shot at a good idea and great score. I didn't have time to try other good ideas. I may still keep going on this competition for more fun, even if it's over. :)

I used really small learning rate 0.005. Combined with XGB pyramid meant I was doing 10 rounds of 200 tree boosted forests to start, and along with a lot in the middle, ending with 0.0025 learning rate for last 9600 rounds. So each single model run was a bit of an ensemble by itself.

Train 5 times on entire dataset vs 5 fold CV didn't seem much different, maybe marginally better, when I tried it with prior model, but I didn't have much time at the end so just went with it. Maybe 10 fold CV would be better, good diversity and still train on almost all data.

I trained two models at the end, one that dropped R_1, S_11, D_59, S_9. I was hoping that days overdue mega-feature would reduce the need for them at step two, and thus eliminate the concern of private LB shakeup on those features. However, it got much worse score on public LB, so I wasn't willing to try it, and did submit my best model in the end.

The next thing I really wanted to try was inspired by reading about a prior credit kaggle competition, I wanted to use KNN to create features.

Especially after reading other people's great ideas, I really wonder if multiple ways of using diverse approaches to extract features from the 13 statements would be ideal.

In other words, for each customer and base feature col with 13 statements, do things like:
* Predict a few different things:
  * 'days overdue' (or predict target, but then you have to be careful with nested folds)
  * P_2
  * predict next in sequence
* Using a few approaches:
  * XGB or LGBM
  * LSTM, RNN or something
  * KNN
  * linear regression
  * predict next in sequence with simple least squares line
* Using data diversity:
  * besides using all 13 statements, what about using 3, 4, 5, and 6 statements to predict *next* on: days overdue, P_2, and next in sequence? This gets a lot more than 1 training sample per customer for training each of those models, and should do well with the "why not both?" approach that throws all features to the model to figure out.

Other thoughts:
* Maybe a good approach would've been to spend a lot more time trying to create any and all aggregate features.... on P_2 only, and see how well I could predict using ONLY P_2? Then create meta features aka mega features like days overdue prediction, and use what worked well on P_2 on that and maybe on all feature columns.

* I probably made a couple interesting mistakes:
  * I should've allowed B_29 in the days overdue model, I think. I forgot to try it.
  * I allowed everything to be float16. I think that might've really hurt on my super feature for days overdue prediction, I should've allowed the top few features (at least) to be float32.
