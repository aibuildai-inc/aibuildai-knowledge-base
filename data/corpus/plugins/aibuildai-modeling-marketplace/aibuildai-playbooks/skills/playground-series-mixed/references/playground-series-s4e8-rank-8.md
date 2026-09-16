# 8th Place Solution with Autogluon🤔

Competition: playground-series-s4e8
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s4e8/discussion/531374

First and foremost, I want to extend my gratitude to all the organizers, participants, and you, the reader! I must admit, there was a bit of luck involved in achieving 8th place. For my model, I simply opted for AutoGluon, but I still want to share and discuss my thoughts with you.

In my first run with AutoGluon, I tried using a GPU, but it didn’t seem to be utilized—perhaps AutoGluon determined it would actually slow things down. In that initial run, due to time constraints, 17 base models and 13 stacked models were trained. The cross-validation (CV) score was 0.985, with a leaderboard (LB) score of 0.98522. Surprisingly, the private score turned out to be 0.98506, which caught me a bit off guard.

Afterwards, I noticed there was quite a bit of noise in the data, such as numerical values and strange words in categorical features that weren’t present in the test set. I assumed that only single-character categorical features were not noise and set the others to NaN, letting AutoGluon handle them (considering tree-based models can naturally deal with NaNs, I figured this might be better than manual imputation).

After making these adjustments, I increased AutoGluon’s time limit and ran it for three days on two Gold 5320 CPUs provided by my school (a big thanks to them!). This time, 19 base models and 22 stacked models were trained, with the CV score reaching 0.9581, the LB score improving to 0.98525, and the private score hitting a personal best of 0.98507.

Later, I discovered that one XGBoost model took two full days to train😅, so I decided to exclude XGBoost (which had a CV score of 0.9846) for a full retraining. This final training took five and a half days, producing 98 base models and 64 stacked models (some stacked models were lost due to cluster issues). The CV score remained 0.9581, the LB score slightly increased to 0.98528, but the private score dropped slightly to 0.98507. I suspect that XGBoost might still have some significance, and that more complex models could have introduced some overfitting.

Finally, thanks again for reading!

P.S. Honestly, given the massive number of samples, I was also surprised by the leaderboard shakeup. What do you all think?
