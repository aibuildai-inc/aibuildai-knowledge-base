# 7th place solution - aka the first of the losers 😄

Competition: predict-energy-behavior-of-prosumers
Rank: #7
Source: https://www.kaggle.com/c/predict-energy-behavior-of-prosumers/discussion/499649

Thanks to Enefit for hosting the competition!

### Target
As many others I worked with two separate models for production and consumption.

At first I noticed that dividing production by installed_capacity and consumption by eic_count improved the performance.

In a second moment I tried, as many other, to predict the differnece with the target 48 hours ago, but I tried to explore the autoregressive settings a bit more in depth.

In particular I considered the general case of an autoregressive model

$$Y(t) = \epsilon + \alpha_1 Y(t-1) + \alpha_2 Y(t-2) + \alpha_3 Y(t-3) + ...$$

The final model is then trained to predict the residuals of this first AR model. This is a generalization of the simpler "predict the delta" approach.

After extensive cross validation on a 5 fold sliding windows schema I ended up the the following setup

For production:

$$taget\\_production = normalized\\_production - normalized\\_production\\_lag48h$$

For consumption:

$$taget\\_consumption = normalized\\_consumption - 0.5 \ normalized\\_consumption\\_lag48h - 0.1 \ normalized\\_consumption\\_lag76h - 0.1 \ normalized\\_consumption\\_lag92h - 0.1 normalized\\_consumption\\_lag120h - 0.1 \ normalized\\_consumption\\_lag144h - 0.05 \ normalized\\_consumption\\_lag168h - 0.05 \ normalized\\_consumption\\_lag336h$$

### Features
The features used were the same as many public notebooks
- county, product_type, is_business + combinations
- hour, day, weekday, month + harmonic features
- eic_count, installed_capacity
- gas & electricity prices
- forecast weather + 7 days lag
- historical weather + 7 days lag
- holidays + days before/after
- releaved target (both production and consumption) with lags (2, 3, 4, 5, 6, 7, 14 days) + normalized, mean, std variations

In total I used 192 features

Stuff that didn't work
- difference between historical and forecasted weather
- more lags
- variance, covariance, slope, kurtosis of other features
- various aggregations of radiation features
- physics formula based on weather data

### Models
I tried to optimize the parameters of LGBM, XGBoost, CatBoost and sklearn's HistGradientBoostingRegressor using Optuna on a 5 fold sliding windows CV schema with a kept aside test set. I also implemented a custom early stopping rule to prune cases that performed badly on the first few CV folds. 

At the end of the process I got a set of candidate models and I studied different ensembling strategies. The best one was to simply average the top 3 models. 

Doubling some models by repeating the training with different seeds was also useful

The final ensemble was the following

- LGBM: best_params x 2 seeds
- LGBM: second_best_params
- XGBoost: best_params x 2 seeds

The same parameters were used for production and consumption, since my experiments lead to the conclusion that a good set of parameters for one was generally good for the other (ie. optuna couln't improve it)

### Training 
The 10 models (5 production + 5 consumption) were trained just once at the start of February

### Post processing

I also tried a few ideas for post processing but I didn't see any improvement in performance
