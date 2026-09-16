# Top 1% solution (28th)

Competition: m5-forecasting-accuracy
Rank: #28
Source: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163199

So I thought about sharing some points of my strategy with you guys.

I decided not to use any multipliers. I'm not a big fan of it and prefer to work on generalizing my model by working on feature engineering and tuning hyper parameters.

## Feature Engineering

I used some ideas published  by @kyakovlev at [https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic](https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic) such as:
- rolling mean and std  - lag [ 28 ] rolling  [ 7 14 30 60 180 ]
- rolling mean - lag [ 1 7 14 30 ] rolling [ 7 14 30 60]
- mean and std target encoding  for 'item id', 'cat id' and 'dept id'

From [https://www.kaggle.com/sibmike/m5-out-of-stock-feature-640x-faster](https://www.kaggle.com/sibmike/m5-out-of-stock-feature-640x-faster) I took
- 'gap e log10' and 'sale prob' and obtained its rolling mean with lag 28.

I have also used features such as last 2 digits of 'sell price', 'days from last price change', 'sell price ratio', 'days from last sale', 'release week', 'weeks from release', 'cluster', 'ADI rolling mean' and target encoding taking 'weekend', 'snap' and 'event' + 'preholidays' into account. 

There are other features as well that don't need mentioning as they were basic ones. I also fixed some outliers I've found and others mentioned at [https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/161718](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/161718) .

About the data, I actually calculated the features using the whole dataset since 2011 except encoding features where I set the lowest date to 2013.

## Model

I used lightgbm and created models for each store, departing from @kyakovlev s model, and tuning it to my needs. 

I trained using a walk-forward cross validation I adapted from [https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8](https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8)

## My thoughts

I think my solution can be improved, a lot. By Sunday, my model started to behave very strangely and I decided to rebuild everything from zero, so I didn't have much time to  analyze which features were indeed good or not. I cut off a bunch of features, and didn't  have time to test Bayesian encoding and other ideas, which I'll do now with more calm.

Hope this helps you all understand some approaches of your fellow competitors, although we saw that the result of this competition could have also been met simply with 'some' multipliers. A big thanks to those that shared their ideas and helped to improve my model.

Let me know if I forgot to mention anyone's idea, and I'll give the proper credit/ referencing.
