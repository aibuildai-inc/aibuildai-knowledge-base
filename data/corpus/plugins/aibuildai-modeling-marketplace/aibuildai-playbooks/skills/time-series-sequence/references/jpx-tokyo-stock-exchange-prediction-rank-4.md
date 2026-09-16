# 4th place solution

Competition: jpx-tokyo-stock-exchange-prediction
Rank: #4
Source: https://www.kaggle.com/c/jpx-tokyo-stock-exchange-prediction/discussion/359151

#### update
We published the code in the following notebook.
https://www.kaggle.com/code/flat831/4th-place-model/notebook?scriptVersionId=100052889

---

Thank you JPX for sharing your data and hosting a fun competition. Thank you Kaggle. Thank you Kagglers for sharing many helpful discussions and notebooks. Thank you my teammates, K-azu and hayato03!

## Model Summary

Our model is very simple. To be honest, this result is mostly luck.

model1(private score: -0.196)
Rank return_1day in ascending order
If ExpectedDividend is greater than 0, make it the lowest.

model2:(private score: 0.347 <- this model is 4th place!)
Rank return_1day in descending order.
If ExpectedDividend is greater than 0, make it the lowest.

return_1day is based on the code in this [notebook](https://www.kaggle.com/code/smeitoma/train-demo).
`return_1day = feats["AdjustedClose"].pct_change(1)`

Below is a brief explanation of the reasons for this approach.

## Difficulty in model evaluation

https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/discussion/320323

As noted in the discussion above, the random model scores can be approximated by a normal distribution of μ = 0 and σ = 0.13785. Therefore, even if we were to create a model with a score of -0.3~0.3, we cannot assume that it is a good model. Furthermore, if we create many models, the probability of a good score by chance increases. (Like the multiple comparisons problem in statistics)
Finally, we gave up on creating a ML model and decided to submit a simple approach to avoid errors.

## ExpectedDividend

We adopted this variable because we were confident that it would be effective.

The definition of ExpectedDividend column is as follows.
> Expected dividend value for ex-right date. This value is recorded 2 business days before ex-dividend date.

And, the definition of target column is as follows.
> Change ratio of adjusted closing price between t+2 and t+1 where t+0 is TradeDate

Thus, since t+2 is the ex-dividend date and t+1 is the date of record, target is likely to be negative.
(However, since ExpectedDividend is almost null, this alone does not get high score...)

## Flipped ranking

https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/discussion/356038

As noted in the discussion above, flipped the ranking and we get a score multiplied by -1. In the competition, we can choose two models, so adopting this strategy will ensure a score > 0.
