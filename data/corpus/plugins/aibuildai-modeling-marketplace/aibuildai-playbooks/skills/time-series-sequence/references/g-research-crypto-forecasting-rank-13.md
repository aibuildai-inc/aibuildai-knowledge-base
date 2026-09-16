# 13th Place (Final) 1st Place (6 Weeks In) Final Solution

Competition: g-research-crypto-forecasting
Rank: #13
Source: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/313386

[View the final submission notebook here.](https://www.kaggle.com/tomforbes/1st-place-6-weeks-in-final-submission)

Final Update - Congratulations to all the winners! While this comp was a bit of a rollercoaster, I'm very pleased to get my first competition Gold. Disappointing to just miss out on the top 10 after being there so long, but thats always possible with this type of challenge.

Original post below.

### Initial Comments

First of all thankyou to G-Research and Cambridge Spark for hosting this competition and to the Kaggle staff for making everything run relatively smoothly. I found it so interesting and more open ended than other comps I've competed in, in my short Kaggler career.

It goes without saying that while I'm in 1st Place now there is no guarantee it will stay that way in the next 6 weeks, but I'm confident I'll be there or thereabouts. Hopefully you find this solution useful regardless of the final outcome.

Elephant in the room - there has been a lot of concern about the validity of the final LB results and requests to release the final test data after the competition ends so that competitors can validate their final scores. I think this would be a good idea and should put any concerns to bed. If it's possible & not too much effort for Kaggle to accommodate the request please consider it. However, as far as im concerned my current score seems reasonable based on all the scores I achieved in testing. I have no reason to believe any of the LB scores are incorrect until there is evidence to the contrary.

**I would encourage those who feel their score is unexpected share their solutions as the Kaggle community might be able to debug any issues/misunderstandings.**

It is obviously difficult to accurately predict crypto prices, but it's possible to find *some* signal - hopefully this notebook will shed some light on how I got mine.

*Disclaimers*
- *this was not my exact final submission but very similar*
- *I have removed the dataset containing my pretrained models and scalers - so some commands will error*

### Summary

- 17 features with lagged and timestamp averages for all models.
- Ensembles of LGBM and Keras NN models.
- Target Engineering and prediction switching.
- Some training on extra cryptos outside competition scope.

Test Scores (no lookahead bias):
- 0.0565 best score on original testing period (Jun 13 2021 - Sep 21 2021)
- 0.0475 best score on supplemental_train.csv update. (Sep 21 2021 - Jan 24 2022)
- 0.0465 best single model score (Jun 13 2021 - Sep 21 2021)

### External Data
I downloaded free public data from multiple exchange apis (Binance, FTX, Coinbase, Kucoin... etc.) to see if this extra data would improve my models. Some of my final models were trained on extra data. Specifically, using some currencies not included in the competition (e.g. XRP, ZEC...) from Binance seemed to provide a consistent small improvement over multiple timeframes.

### Feature Engineering
I settled on a group of 17 features for all models. There are 8 lagged features - **a simple mixture of EMA's, historical returns and historical volatility over various lookback periods** - and these features were averaged across timestamps to produce 8 more. Asset_ID was also included. It was tricky to find a group that performed the best consistently across different time periods, adding and removing features from this set seemed to reduce performance. Im sure this isn't the optimal feature set, but this group seemed to work well enough.

It was also important to perform some kind on binning on the features, especially for training the LGBM model. The commonly used reduce_mem_usage function and some rounding functions seemed to provide a suitable amount of bins. I found binning to 500-1000 unique values worked well for any given continuous feature.

### Target Engineering
I think a crucial part of this competition was manipulating the target. Thanks to the work of **@alexfir** and other Kagglers [we found out how the target was being calculated](https://www.kaggle.com/alexfir/recreating-target).
I thought it would be useful to split the target into two components:
- The forward 15 minute return of an asset
- The beta component, where we calculate the mean of forward 15 minute returns for all assets and incorporate past 15 minute returns for the previous 3750 timestamps

As **@gengdaiziwang** illustrates [in his helpful notebook](https://www.kaggle.com/gengdaiziwang/are-we-training-models-on-an-inaccurate-target), we can see that in the case where an observation is missing for a given asset, the beta component is automatically set to 0.

[[download.jpg]](https://postimg.cc/Dm1B6rXM)

But this makes it a completely different target! A model trained on the target given as standard will have to deal with training on a target that randomly switches between a 15 minute forward return for one asset vs what is essentially a 15 minute forward return *relative to the other assets* - very different things.

The approach I took was to create recreate two targets to represent these two different cases. Then I could train separate models on these recreated targets which would learn the two cases much more effectively.
[See an implementation of this target engineering](https://www.kaggle.com/tomforbes/target-engineering).

Now, having two models trained on different targets, I can alter my predictions to match the target produced by the api. The api provides all the information i need to figure out whether the beta component will be == 0 because if an asset is missing from an iteration i know that for the next 3750 iterations the beta component == 0 for this asset. **Therefore my next 3750 predictions will come from a model that was training on a target constructued without a beta component.**

This target engineering and dual model method added roughly 0.01 to my score on the original test period, although this did vary quite significantly for other testing periods.

To summarise as simply as possible:

- TargetZero = A target based on 15 minute Forward Return only
- TargetBeta = TargetZero + Beta Component
- ModelZero = Model(s) trained on TargetZero
- ModelBeta = Model(s) trained on TargetBeta

[[gres.png]](https://postimg.cc/xJJFnBqf)

### Models

I found LightGBM worked well and was easy to experiment with, i used the weighted correlation evaluation metric and fairly out of the box hyperparameters, tuning didnt seem to add much and wasn't consistent across time periods.

I also used a Keras NN heavily influenced by **@lucasmorin's**
[excellent notebook](http://https://www.kaggle.com/lucasmorin/online-fe-nn-loss-shap). This just worked really well for me out of the box and i couldnt find many ways to improve the architecture or hyperparameters.

Ensembling these two models worked well and added an extra 0.005 to my score (roughly) although the improvement varies in different market regimes. I also experimented with ensembles using catboost and XGboost but the predictions from most gradient boosting models were too highly correlated to provide much ensembling benefit. I found the diversity in predictions between NN and GB were a good match.

I used a fairly simple CV method for training and testing. My folds were based on timestamp values similar to the below:
- Training Fold 0: 1514764860 - 1570000000 
- Training Fold 1: 1514764860 - 1580000000 
- Training Fold 2: 1514764860 - 1590000000 
- Training Fold 3: 1514764860 - 1600000000 
- Training Fold 4: 1514764860 - 1610000000
- Validation Fold 0: 1570000000 - 1580000000
- Validation Fold 1: 1580000000 - 1590000000 
- Validation Fold 2: 1590000000 - 1600000000 
- Validation Fold 3: 1600000000 - 1610000000 
- Validation Fold 4: 1610000000 - 1620000000 
- Test Fold A: 1615000000 - 1623542400
- Test Fold B (Supplemental Train Update): 1623542400 - 1643000000

Since this public lb was useless in this competition, it was essential to construct a robust CV framework to have a reliable benchmark for making improvements to models. It was also important to have test folds in several different market regimes to avoid overfitting to one period.

**My best single model was a NN trained on fold 0 data only (surprisingly).**

I was fairly selective on picking models to include in the final submission as some folds just trained much better than others and had more consistent outperformance on unseen data.

### Submission

This is almost the exact notebook I submitted for my final predictions. I used lists to store historical data for feature calculations and did most calculations using numpy or lists, avoiding pandas at all costs.

The submission completes in roughly 7 hours.


### What didnt work
- Layering extra data from different exchanges for the same asset. E.g training on BTC data from Binance + Coinbase + FTX + G-Research version. Didnt harm but didnt improve models significantly either.
- Framing as a classification problem - poor results
- Hyperparameter tuning LGBM
- PCA features
- Using all 14 assets features per row / predicting on 14 targets.
- Meta modelling GB and NN predictions

### With more time id try
- Extra Feature engineering
- Pytorch implementation of keras NN for ensembling
- LSTM
- Other model architectures
