# 39th Place - Solution Overview & Code

Competition: jane-street-market-prediction
Rank: #39
Source: https://www.kaggle.com/c/jane-street-market-prediction/discussion/224029

I thought I’d post an overview of my solution while I am relatively high on the leaderboard which might not last for long. My solution is based on a simple MLP with some elements that I have not seen posted before including a bit of feature and target engineering and weight-dependent voting in post-processing.
  
# 1.	CV Strategy

For me this is by far the most important part – with the amount of noise in the data everything else would not be possible to evaluate without a decent CV strategy.

I’ve spent a huge chunk of time trying to find a validation approach that gave at least somewhat consistent and logical results. I eventually landed on using mean OOF utility scores similar to what I posted [here](https://www.kaggle.com/dmitryvyudin/jane-street-oof-utility-scores). From the very start I decided not to use the public LB other than to check that the code runs and does not time out – otherwise the temptation to overfit the public LB would be too difficult to resist. 

I used all the data for both training and validation based on `GroupKFold` split using dates as groups – this was a risky decision as I assumed there was no leakage between training and validation dates. I have a few reasons to think that this leakage indeed was not a problem:
•	First, this seems to be an HFT competition – as @gregorycalvez noticed in this excellent [notebook](https://www.kaggle.com/gregorycalvez/de-anonymization-time-aggregation-tags), the `NaN` patterns suggest that the data contains some rolling intraday features. This does not mean that there are no other rolling features covering many days, but from my point of view, it does make it less likely/important.
•	I tried splitting data into groups of consecutive days – from 1 to 100 days in one group. In case of leakage you could expect to get better results with 1-day groups compared to 100-day groups because there is potentially much more interaction between training and validation data. But after tons of experiments I just could not see this effect.
For the actual training I used groups of 50 consecutive days.
•	At the beginning I tried splitting data completely randomly, i.e. using trades from the same date both for training and validation. In this setup the intraday leakage was noticeable but even then my model had to run for a lot of epochs to pick up this information with validation scores continuing to improve even after 100’s of epochs. I have not observed this behavior with `GroupKFold`.

# 2.	Feature Engineering

There were a number of discussions about the meaning of `feature_0` (buy/sell, long/short?). I have no idea what the correct answer is – my hypothesis is that it is produced by a separate JS model that selects the trading opportunities. And if this is true then its recent history might be indicative of some sort of market condition. This gave me an idea to build a series of rolling 'lag' features based on `feature_0` and this resulted in a modest but noticeable improvement in the CV scores.

I have also added a few other features based on the ‘clock’ `feature_64` that together also improved the CV score:
•	Binary feature representing part of the trading day (before/after lunch)
•	Number of trades suggested by JS algorithm earlier today (for the first part of the day) or after lunch (for the second part of the day) - the intuition here that together with 'clock' this feature could also represent a market condition (e.g. more trade opportunities = more volatility)
•	'Gradient' of `feature_64` with respect to timestamp - similar intuition to the previous point

# 3.	Target Engineering

Like many others, I have noticed that treating this task as multi-label classification leads to better results compared to trying to predict just one label - `resp`. I have made a couple of adjustments compared to most public notebooks:
•	I did not use `resp_4` - my CV always went down when I tried to add it. This could be to some extent explained by the fact (conjecture?) that the time horizon of `resp_4` is longer than that of `resp`.
•	Instead, I added the mean value of `resp`, `resp_1`, `resp_2` and `resp_3` as a separate target which did improve the CV score. This can be thought of as a proxy for the general direction of returns over the whole `resp` time horizon.

# 4.	Model & Training

Nothing interesting architecture-wise – very basic 3-layer MLP with batch normalization and dropout. 

For optimization I used [LAMB ](https://arxiv.org/abs/1904.00962) with Lookahead – according to the paper, LAMB is supposed to work well with large batch sizes and it seemed to outperform other optimizers I tried. 

# 5.	Inference & Post-Processing

I did not like the idea of playing with the threshold – values other than 0.5 seem artificial and lack any intuitive meaning. Instead, for most of the competition, I used a ‘qualified’ majority voting. I.e. only accepting the trade opportunity if 66% of models ‘voted’ for it. 

Later on, I started taking the weight of the trading opportunity into account - the more the weight of the opportunity, the more confident I have to be to act on it. This results in slightly better utility scores since the utility score formula punishes high variance.

Overall I used 50 models (5 folds & 10 seeds) with each model having 5 votes – one vote per target.

# 6.	Code

•	Training - https://www.kaggle.com/dmitryvyudin/jane-street-tf-lamb
•	Inference - https://www.kaggle.com/dmitryvyudin/jane-street-weighted-voting-inference

# 7. Other

Some of the things I tried but could not make work based on my CV strategy:
•	Treating the problem as regression as opposed to classification – much better results with L1 loss than MSE but still worse than classification
•	Using weight or log(weight) in the loss function
•	Adding noise to targets
•	Knowledge distillation as a target de-noising technique – this looked very promising but I found a leakage literally hours before the submission deadline and had to revert to earlier, simpler models
•	[AdaHessian ](https://arxiv.org/abs/2006.00719) optimizer (was very excited about the results initially but in the end LAMB performed better)
•	TabNet – found it too slow for this competition
