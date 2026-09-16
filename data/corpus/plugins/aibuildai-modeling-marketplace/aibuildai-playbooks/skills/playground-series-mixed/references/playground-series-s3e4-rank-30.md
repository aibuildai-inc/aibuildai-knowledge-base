# 30th Place Solution

Competition: playground-series-s3e4
Rank: #30
Source: https://www.kaggle.com/c/playground-series-s3e4/discussion/382447

This comp was an interesting one, and for me the hardest of the Playground Series this month. The addition of the time component to the data, combined with how the train and test sets were split by time, made me more unsure of how to best build a local CV schema and unsure of its reliability. In the end, I used sklearn's **TimeSeriesSplit** for CV, and at the very least it seemed fairly reliable for optimizing hyperparameters.

My final solution was pretty simple, just an average of 50 **XGBoost** models with the same hyperparameters, trained using Stratified KFold with 5 folds and 10 repeats. I ended up using the entire train dataset plus the instances of fraud from the original Credit Card Fraud data.

There were 2 things that I think helped my score a little. First, I added **transformed versions of the V features created by subtracting the daily average for that feature**. This was inspired by @paddykb's [comment ](https://www.kaggle.com/competitions/playground-series-s3e4/discussion/381415#2116636) about how many features behaved differently between day 1 and day 2, and upon further observation, it seemed to that the general up-down trend of the features throughout a day was similar between day 1 and day 2, and the main difference was that this trend was shifted up or down between days. So I figured that subtracting out the daily average might help uncover some more signal. This did not improve my Public LB score, but it gave me a boost of **+.0015** on the Private LB score.

The second addition was inspired by @siukeitin's [comment](https://www.kaggle.com/competitions/playground-series-s3e4/discussion/381483#2117170) about using the V features to identify customers. Out of curiosity, I ran a check to see if grouping the train data by any 2 V features would result in pure groups of all fraud or all non-fraud, and I found that the combination of **V14 and V21** did just that. This gave me 466 rows in the test set that had a (V14, V21) that also existed in the train set, and based on the train data, I predicted 0 for those rows. This boosted both my Public and Private LB scores by **+.0004**.

Sadly, it was my decision to include the instances of fraud from the original data that prevented me from placing higher, and I suspect that anyone who scored .83 or above probably did it without any original data. Had I excluded the original data, I would've scored .8333, enough for 1st place. With all of the well-documented differences between the original and synthetic data this time around, my instinct was telling me to not use the original data, but it was hard to go against what my CV and the Public LB were telling me. A good lesson for the next one.
