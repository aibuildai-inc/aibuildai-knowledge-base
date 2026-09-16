# ~5th Place Simple Linear QuantReg Solution

Competition: covid19-global-forecasting-week-5
Rank: #5
Source: https://www.kaggle.com/c/covid19-global-forecasting-week-5/discussion/154773

The task here is a time series forecast to predict Covid19 daily numbers of Cases and Fatalities for countries and also for some US regions. But they also challenge us to predict the 5% and 95% quantiles of that numbers and the metric chosen for that is the Pinball loss and optimizing it isn't an easy task. Pinball loss guarantee that all forescasts are between 5 and 95 quantile range, every forecast outsite that limits is extremely penalized, so models needs to take care when predicting the lower and upper limits.

For this challenge I build a simple linear model using [QuantReg](https://www.statsmodels.org/devel/generated/statsmodels.regression.quantile_regression.QuantReg.html) regressor from statsmodels lib. QuantReg showed to be very good to estimate the quantiles forecasts out of the box. 

To train the model I used data from 2020-04-01 to 2020-04-26 and to validate from 2020-04-27 to 2020-05-10. Note that I used 2 weeks only for validation and the Private LB will be calculated using 30 days, so I expect some drifts in my forecasts for the last 15 days of private data.

Were trained two independent models, one for Daily Cases and one for Fatalities. Both models uses similar features build from some lags and rolling windows statistics. Due the quantile bands forecast, it is logical that a good feature could be the rolling window of the standard deviation. I added rolling windows of std() with windows size 7, 14, 21 and 28, but due a typo in my code only windows 7 and 28 are being used in the final submission. I tried other kinds of rolling windows functions but everything else decreased my validation scores.  All my predictions in the validation period and in Private dataset are made recursively  (prediction for day t becomes the lag(1) of day t+1).

Also to avoid issues with the 5 and 95 quantiles close to the end of the 30 days period, I added a linear margin every day in order to always increase the 95 quantile and decrease the 5 quantile forecast.

The validation score for daily Cases is 0.2619 and for Fatalities 0.2341,  ~0.248 in average for the two weeks validation period. Right now, about 15 days of Private data are already scored and my Pinball score is 0.2301, pretty close of my calculated scores. 

Here is an example of my forecasts for US Cases and Fatalities:




Link for my notebook [here](https://www.kaggle.com/titericz/test-1?scriptVersionId=33803268).
