# [3rd place] Summary of my work

Competition: mlb-player-digital-engagement-forecasting
Rank: #3
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/256620

***2021-09-10 updated***
The final ranking is 3rd! Thank you all :) My source code is a bit messy, but you can see it here: https://github.com/nyanp/mlb-player-digital-engagement and https://www.kaggle.com/nyanpn/3rd-place-solution-inference-only .

---

First of all, I would like to thank the hosts for organizing this competition.
It was a very tough competition but I learned a lot from it.

I have no idea what my final standings will be, but I will share with you what I did.

## Model strategy
Lag features are very useful in this competition, but we cannot use target information in the test data period. So we need to set an appropriate "gap" between the prediction date and the lag feature data.

For example, if you want to make a prediction for 8/4, 
you need to create a gap of at least 3 days because you cannot use the target information from 8/1 to 8/3.

I divided the test data into 8 periods and used different models with different gaps in each period (gap: 0, 3, 7, 14, 21, 28, 35 ,45days).

[https://raw.githubusercontent.com/nyanp/mlb-player-digital-engagement/main/docs/img/time%20series%20gap.png]

## Model
The three models were ensembled with different weights for each target.

- LightGBM
- MLP
- 1DCNN

1DCNN is the same as [the 2nd place solution of MoA Competition](https://www.kaggle.com/c/lish-moa/discussion/202256).

### Validation
Time based split.
- validation before update: 2019/8, 2020/8, 2021/4
- validation after update: 2020/8, 2021/6, 2021/7

Basically, I only adopted ideas that improved the score in all periods.

## Features
I used ~440 features. In addition to joins and asof merge of basic tables, the following features were used:

- lag features per player
    - Average of the last 7/28/70/360/720 days
    - Average over the on-seasons
    - Average for the same period in the previous year
    - Average of days with/without a game
- number of events, pitch events, action events
- days from last rosters, awards, transactions and box scores
- sum of box scores in the last 7/30/90 days
- number of games and events in the day
- event-level meta feature
    - aggregation of predictions of model trained on event table
    - group by (date, playerId), (date, teamId) and (date)

### Cumcount Leakage
There is a strange correlation between the cumcount of the dataframe retrieved from the Time-Series API and the target.

I noticed this problem 3 days before the competition ended. I did not post it in the discussion as it might confuse the participants, but contacted the host immediately.
Adding this cumcount to the features only improves the CV a little bit, so it's probably some kind of artifact or something, but even if it doesn't improve the CV much, it's better to shuffle the test data since it's nonsense that the order of the rows makes sense.

I did not end up using this leak for final submission.

### Implementation Note
Building a complex data pipeline in Jupyter Notebook with the Time Series API can be a big pain. I'll share some of my efforts.

- Maintain the source code on GitHub and paste the BASE64-encoded code into the jupyter notebook
    - see: https://github.com/lopuhin/kaggle-imet-2019
- The inference notebook is also maintained on GitHub and automatically uploaded as the Kaggle Kernel through GitHub Actions
- Avoid the use of pandas and instead use a dictionary of numpy arrays to manage state updates
- Use the same feature generation function for training data and inference
    - Both training and test are treated as streaming data, and features were generated using for-loop.
    - This is the most important point to get a stable and bug-free data pipeline
    - see: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/196942
- Debug code locally using the API emulator
    - Test the robustness of my inference pipeline by "dropout" some of the data returned by the emulator (a kind of "Chaos Engineering")
- Catch exceptions in various functions and convert them to appropriate "default" values

Thanks to all of this, I was able to finish the competition 1st stage without making a single submission error.
