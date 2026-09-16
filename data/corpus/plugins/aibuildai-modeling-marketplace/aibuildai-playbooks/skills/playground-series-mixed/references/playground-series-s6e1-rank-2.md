# 2nd place - NNs sometimes work better than GBMs

Competition: playground-series-s6e1
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s6e1/writeups/2nd-place-nns-sometimes-work-better-than-gbms

Congratulation to all the winners and thanks to those contributing quality discussions and code: @include4eto, @omidbaghchehsaraei, @cdeotte, @siukeitin, @mikhailnaumov. If you haven't read them already, I suggest you visit these discussions:

- https://www.kaggle.com/competitions/playground-series-s6e1/discussion/665915
- https://www.kaggle.com/competitions/playground-series-s6e1/discussion/665965
- https://www.kaggle.com/competitions/playground-series-s6e1/discussion/669056

My solution is conceptually identical to what I already published [**back in October**](https://www.kaggle.com/competitions/playground-series-s5e10/writeups/1st-place-i-think-it-was-genetic-programming): make as many diverse models as possible and combine them all in a large ensemble. Here I will only describe the two major differences that were unique to this competition.

First, in my hands TabM worked by far the best when it came to individual models. It wasn't even close when it was compared to GBM models, even though I didn't try LightGBM that much. But I made many XGB and CatBoost models that were inferior to TabM, even when the same starting data was used. **This is yet another proof that there is no absolute superiority of GBMs over NNs in tabular competitions.**

Second, I benefited enormously from feature engineering strategies published in November by @mahoganybuttstrings and @angelosmar1 - thank you both! I urge you to revisit their solutions:

- https://www.kaggle.com/competitions/playground-series-s5e11/writeups/1st-place-a-lot-of-features-a-lot-of-models-an
- https://www.kaggle.com/competitions/playground-series-s5e11/writeups/2nd-place-solution-7-models-but-1-was-also-enou

Speaking of @mahoganybuttstrings, congratulations on yet another absolutely dominant win! Anyway, I have zero talent when it comes to rational feature engineering, but I know how to spot good features when I see them. There were plenty of them in the two solutions above, and I added them gradually and in various combinations to the tune of 60 TabM models that eventually were included in my 75-model ensemble. If you are wondering how I managed to make that many models: 1) it was various combinations of engineered features, ranging from ~170-700; 2) those features were combined with 6 or so hyperparameter sets that were found by tuning; 3) in addition to conventional modeling, I also did predictions over residuals from linear equations. Finally, I also added 5 features found by genetic programming that were meant to recreate the formula that generated the original dataset. Eventually this became 14 GP features that were used for modeling. The best two equations, which were used for predictions over residuals, are listed below. They create features with RMSE 8.9703 and 8.9741 to the target, respectively.

```
           formula = 5.760845*X['study_hours'] +
           4.760845*(X['sleep_quality'] == 'average').astype(int) +
           8.760845*(X['sleep_quality'] == 'good').astype(int) +
           9.52169*(X['study_method'] == 'coaching').astype(int) +
           0.325220702899505*X['class_attendance'] +
           (X['study_method'] == 'group study').astype(int) +
           4.760845*(X['study_method'] == 'mixed').astype(int) +
           5.760845*(X['facility_rating'] == 'high').astype(int) -
           (X['facility_rating'] == 'low').astype(int) +
           2*(X['facility_rating'] == 'medium').astype(int) +
           X['sleep_hours']

           formula = 5.87666929874987*X['study_hours'] +
           5*(X['sleep_quality'] == 'average').astype(int) +
           9.674877*(X['sleep_quality'] == 'good').astype(int) +
           8.222835*(X['study_method'] == 'coaching').astype(int) +
           0.327161004763791*X['class_attendance'] +
           3*(X['study_method'] == 'mixed').astype(int) +
           3.659515*(X['facility_rating'] == 'high').astype(int) -
           4.222835*(X['facility_rating'] == 'low').astype(int) +
           X['sleep_hours'] +
           2.020001

```

Out of 75 ensemble models, all but 7 were NNs of some kind, and there were about 20 more TabM models that didn't make it. Best individual models:

| Model | CV | Public LB | Private LB |
| --- | --- | --- | --- |
TabM | 8.590414 | 8.55355 | 8.59254 |
| XGBoost Optuna | 8.605121 | 8.56390 | 8.60570 |
| Keras FM | 8.658336 | 8.58030 | 8.61956 |
| xLearn FFM | 8.707240 | 8.67284 | 8.70387 |

As always, my goal was to select the best solutions, and indeed that was the case. That was made easier by good CV-LB correlations, but there was still enough difference in scores that I hedged my bets on two ensembles that had 75 and 71 models. I submitted my best solution 25 minutes before the deadline, and that's usually how close I cut it with all my deadlines.
