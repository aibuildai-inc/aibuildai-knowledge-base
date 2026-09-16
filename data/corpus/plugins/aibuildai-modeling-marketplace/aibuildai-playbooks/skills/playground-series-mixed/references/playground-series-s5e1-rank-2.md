# 2nd Place - Stacking Transformer and Linear Regression

Competition: playground-series-s5e1
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s5e1/discussion/560549

# Forecasting Challenge
Wow, i'm excited to win 2nd place and I feel lucky. Forecasting competitions are hard because choosing the right multipliers for the future **years** (2017, 2018, 2019) has a greater effect on your final LB score and rank then having a model which predicts **days** accurately (i.e. what happens on Monday versus Sunday, or holiday versus non-holiday). In this comp, the training data can help us predict **days**, but what occurs in future **years** requires guessing and/or assumptions.

In this competition we are given sales data for years 2010, 2011, 2012, 2013, 2014, 2015, 2016 and we must predict 2017, 2018, 2019. In each future year, we must predict 90 numbers per day for each country, store, product combination. After we build a linear regression model (with sinusoidal engineered features) and include the effects of: GDP, store, product, country, day of week, day of year, there is still an unaccounted for trend (how data changes per year) in the data (pictured below). 

Below is an image of prediction error from top public notebooks. The y axis is the multiplier we need to multiply our predictions (i.e. percentage error) to match ground truth. We see this error ranges between plus and minus 6%. So what do we do about the future? 



* We can multiple all future predictions by the last known percentage error which is `m=1.06` (like popular public notebooks). 
* We can use no multiplier, `m=1.00`.
* We can use linear multiplier that increases or decreases as we move forward in time `m = 1.06 + slope * (year - 2017)` for some slope. 

These different options are pictured above with dotted lines. There are many choices for future multipliers. This is why forecasting is so difficult. We cannot say with any certainty which future trend is correct (without having more information from outside the train data). So we just need to guess. For my final two submissions, I chose `constant 1.06` (green dotted line) and `mild linear up` (orange dotted line) 🤞

# Transformer Only - Public LB = 0.04867, Private LB = 0.04967 (59th Place)
Using only my public starter [here][1], we can achieve `public LB = 0.04867`, `private LB = 0.04967` and `59th place private` with the following changes:
* train 1 model on all 5 products (for 15 epochs cosine schedule)
* add 30 boolean features for 30 holidays
* use first predictions (of 2017,2018) as pseudo label to train second predictions
* use second predictions (of 2017,2018,2019) as pseudo label to train third predictions
* use the median of 5 models trained with different seeds (for 1st, 2nd, 3rd predictions)
* submit the 3rd predictions
* use no multiplier. Transformer determines what to do about future

Note that we use 2 rounds of **pseudo labeling** (which is in addition to autoregression). For more details about pseudo labeling reading comment below [here][4]. We also ensemble 5 copies of the model with itself trained on different seeds. Both these techniques improve accuracy and help us get good predictions far into the future at year 2019 private test data.

# Linear Regression Only - Public LB = 0.04733, Private LB = 0.04650 (6th Place)
Starting with Konstantin's great public notebook (model 1) [here][2], we can achieve `public LB = 0.04733`, `private LB = 0.04650`, and `6th place private` [here][3] by adding the effect of holidays. (See all holidays [here][3]).
* add country holidays
* keep multiplier `m=1.06`

Below is an example of how to locate and add holidays for a specific country. Holidays can change day each year, so we perform EDA and put black vertical lines before and after a holiday (for a specific country). Then we look at each year 2010 thru 2016 in a specific country and see if the sales consistently are raised or lowered. If so, we add this holiday to our model.

For Singapore we observe that sales are raised for the following 7 holidays each year: Chinese New Year, Easter, Vesak Day, National Day, Eid al-Fitr, Deepavali, Eid al-Adha. These 7 holidays are shown for years 2014, 2015, and 2016 below. For more examples, see my notebook [here][3]. For linear regression model, we boost these windows of time. For transformer model, we add boolean features so the model can find and predict these holidays.


# Stacking - Public LB = 0.04526, Private LB = 0.04498  (2nd Place)
By stacking my transformer over linear regression to predict the residuals (error), we can achieve `public LB = 0.04526`, `private LB = 0.04498`, and `2nd place private`! 🎉 (Stacking versus Ensemble is explained in detail [here][5])

The transformer learns patterns that the linear regression model does not learn. The most efficient way to use both is to train the transformer on the prediction error of the linear regression model. So, first we use the linear regression model to predict the train data Jan 2010 thru Dec 2016. We then subtract the predictions from the ground truth to get the prediction error. Next we train the transformer to learn and predict this error (i.e we train transformer with `target = truth minus prediction` for Jan 2010 thru Dec 2016). Finally we submit the sum of the two models' predictions. 

[1]: https://www.kaggle.com/code/cdeotte/transformer-starter-lb-0-052
[2]: https://www.kaggle.com/code/kdmitrie/pgs501-model-1-time-series-decomposition?scriptVersionId=218663573
[3]: https://www.kaggle.com/code/cdeotte/6th-place-add-holidays-lb-0-046
[4]: https://www.kaggle.com/competitions/playground-series-s5e1/discussion/560549#3113319
[5]: https://www.kaggle.com/competitions/playground-series-s5e1/discussion/560853
