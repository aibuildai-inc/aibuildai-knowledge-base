# 14th Place Solution for the Optiver - Trading at the Close Competition

Competition: optiver-trading-at-the-close
Rank: #14
Source: https://www.kaggle.com/c/optiver-trading-at-the-close/discussion/462653

This post outlines our approach which got us 14th on the private leaderboard

Firstly, thanks to @ravi20076, @mcpenguin, @madarshbb, @cody11null for the collaboration, and Optiver for organizing 

Also, thanks to @wenxuanxx, @zhangyue199, @lblhandsome, for upholding the Kaggle spirit and sharing your feature ideas and notebooks

**Context**
=
Business context: https://www.kaggle.com/competitions/optiver-trading-at-the-close
Data context: https://www.kaggle.com/competitions/optiver-trading-at-the-close/data

**Overview of the approach**
=
To begin with, we fixed bugs in the public notebook, and made a small change to our local validation such that it does not have data leakage when handling global features

```
df['mid_price_movement'] = df['mid_price'].diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
to
df['mid_price_movement'] = df.groupby(["stock_id"])['mid_price'].diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
```
(Btw this fix improved CV but worsened public LB score. Idk why LOL)

Also, we wrote our own functions for RSI, MACD and bollinger bands indicators as the public one gave inaccurate results in inference

**What worked (impact on CV)**
- Features based on revealed targets. We used lags 1,2,3 for target grouped by stock_id and seconds_in_bucket as features into our models (-0.005)
- Signed representation of imbalance_size (-0.003)
- Continuous model training with revealed targets. We refitted our LGB and CatBoost model at fixed intervals (elaborated later) (-0.006)
- Performing CV in streaming fashion. To do this, we saved the data each time period in <date_id>_<seconds_in_bucket>.csv and delivered them one by one in chronological order when calculating CV. This took much longer but correlated better with the public LB
- Zero sum post processing (-0.005) (but we are not sure in private LB so we only chose 1 submission with this)
- Global Features (-0.004). We have to re-initialize these values this whenever we want to retrain the model to keep it up to date
- Technical indicators RSI, MACD and bollinger bands. We had to rewrite them for good results (-0.002)

**What did not work (impact on CV)**
- Group by date_id and stock_id for rolling features instead of stock_id (+0.003). We didn’t do this in the end
- Lagged features shift(x) where x is large. Made CV better but worsened LB
- Rolling features over a window x where x is large. Made CV better but worsened LB
- Sector features (+0.002)
- Neural Networks
- Triplet Imbalances (+0.001) realised this feature gave very unstable values because of precision issues so we decided to discard this even though they improved the public LB score (worsened from 5.3315 to 5.3327)
- Dropping features based on feature importances
- Zero mean postprocessing. It makes our ensembles and LB/CV correlation worse for some reason so we didn’t choose this. In fact the ensemble does worse with zero mean (5.333) compared to zero sum (5.3327)

Some additional discussion points of the approach are outlined [here](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/485985)

**Details of the submission**
=
Overall, after much consideration, the submissions that we chose were these two. Thank you @cody11null for tuning the parameters and testing it with your huge 91 model script

170 features, no postprocessing. Public 5.3384. Private 5.4457. LGB + CatBoost
Refitting strategy (assuming day X is the first day where currently_scored is True)
- Day X: Refit LGB, CAT
- Day X+6: Refit LGB
- Day X+12: Refit CAT
- Day X+18: Refit LGB
…
- Day X+54: Refit LGB
- Day X+60: Refit CAT

193 features, zero sum postprocessing. Public 5.3327. Private 5.4458. LGB + CatBoost
Refitting strategy (assuming day X is the first day where currently_scored is True)
- Day X: Refit LGB, CAT
- Day X+9: Refit LGB
- Day X+18: Refit CAT
- Day X+27: Refit LGB
…
- Day X+54: Refit CAT

At each time period, we use the latest LGB and latest CAT and submitted the average prediction of the 2 models

We tried to use the stock weights for a while but got submission scoring error when refitting so didn’t proceed 😢
