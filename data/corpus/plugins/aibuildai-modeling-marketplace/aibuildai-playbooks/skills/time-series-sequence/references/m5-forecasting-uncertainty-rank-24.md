# 24th position solution - with code

Competition: m5-forecasting-uncertainty
Rank: #24
Source: https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/163640

# Global thoughts
I was aiming for gold and i’m extremely disappointed for not getting it. When seeing topics such [as this one](https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/163368), and people with basic solutions getting top ranks, i’m jealous. Due to private datasets was different, models with wider quantiles might have been advantaged. 

At the same time when reading approach like [1st Place Solution](https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/163368) I can only admit they are better, and avoid being a sore loser :) 

Anyway, since few people published their codebase, benchmark and that i’ve a different approach than most people, i think it might be worth sharing it with you. I was expecting to publish a more detailed write up with plots (publish my first paper maybe?) and since not getting gold I’m demotivated :) 

I also spent 1.5 months optimizing pinball loss in the wrong direction (&lt; instead of &gt;=). It was not really clever.

After writing this I just submitted my score with a 0.97 adjustments (didn’t try other adjustments). It would have placed me first place. I’m even more frustrated ^^


# Code
[My gitlab repository.
](https://gitlab.com/JacquesPeeters/m5-forecasting)
# Bias in the data 
By plotting the number of active items over time, we can see that the global growth is not due to growth of walmarts stores’ but probably due to the bias on how the dataset was constructed. 

Therefore when forecasting an aggregated serie, actually it was not historical total sales of the series but the sales of selected products active at that time. 

Interestingly total sales are going higher but average sales per product is going lower. It can probably be explained by survival biais. Items sold from multiple years/a long time are only “survivors'' because they are BestSellers. Items which had lower performance in early years (2012/13) were killed and replaced with newer products in our dataset. 

# Data augmentation
A date can be predicted from different horizons (between 1 and 28). Each time series is small data therefore I expected significant better results with data augmentation. 

Note : I performed this only for granularities not related to item’s.

# Data downsampling
I assumed item granularities can be seen as a pointwise approach. For performance reason I allowed myself to downsample these granularities by half. 
We prefer to sample randomly rather than removing part of the data (between two dates) because we assume that unexplained variance is mostly due to seasonal factors rather than item’s behaviour.

# Datasets
I often focus a lot on FE, during this competition i wanted to focus more on the benchmark part and software engineering first, then on FE. I still performed a lot of FE. 
[You can find a medium post where I describe my philosophy about horizontal scaling if you are interested in. ](https://medium.com/manomano-tech/a-framework-for-feature-engineering-and-machine-learning-pipelines-ddb53867a420)

Note : When hierarchical adjective is used, it means that features are “propagated” to lower granularities. Eg : event_trend of total_id can be added to category_id granularities. 

## Weather_[state_id, total_id]
I’m tried to unbiased past sales data with temperature from the date from we are predicting / average temperature from the past. Did not seems to help much. I was not using temperature from predicted dates and therefore complaint with rules. 

## Fe\_sales\_historical
Historical average sales of the serie on differents time windows [3, 7, 14, 28, 56, 112]
Historical quantiles [the 9 quantiles to be predicted, target encoding in a way] on different time windows again [28, 56, 112]
Historical average sales of the serie on the same day of week on windows [1, 2, 3, 4]
Historical average prices multiples windows

##&nbsp;Dayofyear\_trend\_hierarchical
Out-of-fold (each year is considered a fold) target encoding of the average trend of date\_to\_predict / day\_of\_year of date\_from

It was funny to see the trend of hobbies categories pre/post christmas. 

Predicting date t depends of :
    1) on which days of the month historical sales were computed (including the first saturday of the month have a huge impact)
    2) the horizon of the forecast (do we predict the first or last saturday of the month)

It is a 2-D variable problem which is extremely difficult to deal for trees. This trick makes it 1-D.

## Yearly\_trend\_day_hierarchical
Same as previous but based on [date_from_month, date_from_day, date_month, date_dayofweek, date_month_week] which is a bit different and expected to be a bit more robust. 

## Monthly\_trend\_day\_hierarchical
Same but based on day of month

It was sad to see the end of the month was way more extreme in Wisconsin &gt; Texas &gt; California and that it is correlated to GDP … So from data we know that more people in Wisconsin struggle at the end of the month.
Number of days since last sales
Didn’t help much

## Event modeling
The fun part, greatly help local validation.

Use a two-stage kind of causal inference approach. 
* Train a model with all features available. 
* Compute out-of-fold median trend of an event trend = pred/real

Median was preferred over mean, due to the lack of data, i felt it was more “robust” but it is subjective choice

However, there are multiple events on the same date. We need to rank events and assign one and only one to a given date. To do so the same approach is used : 
* Train a model with all features available
* Find event with strongest absolute trend abs(1 - pred/real)
* Assign strongest event to the date 

This trick allowed me to detect that Fathers’ day have a strongest impact than NBA finals therefore was prioritized

Find below the more extreme events for serie total_id.



Edit : i introduced a bug while moving from mean to median this w-e, and it was not out of fold...

# Training

## Hyper-parameter tuning
Due to the amount of models trained, it is difficult to optimize each approach. For a lightgbm approach a single parameter is over-written, namely the number of observations set-up to 100. Given we have a data-augmentation of 28, 5 years of data, we want to avoid overfitting events with single leafs. Eg : creating one leaf for each New-years year. 

## Normalizing
Once again as the item's granularities can be seen a pointwise approach, normalizing doesn’t make sense. On the opposite side, time series approach benefits from it. 

I normalized by sales_mean\_[14, 28, 56, 112]. Only variables related to sales and the target were normalized. Variables related to price weren’t.

## Train/validation split - Retraining on full data
Last XX days of data was used for validation. 

Training/validation was used to train and predict on out-of-fold_validation set. This set was used for stacking. Validation/out-of-fold_validation sets were randomly sampled and not in a timewise fashion. I might be over-fitting because of this random sampling.

Retraining on full data was natural due to the nature of the competition. 

The train/validation training was used solely for validation purposes, out-of-fold stacking, analysis of features importance, and finding the right number iterations of LightGbm models.

## Benchmark of approaches
Due to the fact that multiple solutions were possible, my initial goal was to set up a benchmark framework for testing the maximum of them. What i tried : 
* LightGbm Quantile regression 
* Tensorflow quantile regression
* Regression_l2-to-quantile (regression l2 then quantile regression on top of predictions) 
* Tweedie-to-quantile regression (tweedie regression then quantile regression on top of predictions)
* Ngboost-distribution (compute quantile based thanks to prediction + standard deviation) but it was way too slow and not good

Each approach performs differently on granualirities.

Quantiles from Regression_l2-to-quantile were learned on validation set, and not from training set nor out of fold. Therefore I might be overfitting there :( 



Overall performance after stacking. 


There are huge improvements, but I think a bit over-estimated due to having a random validation/out\_of\_fold validation split as already explained.

See you next time!
