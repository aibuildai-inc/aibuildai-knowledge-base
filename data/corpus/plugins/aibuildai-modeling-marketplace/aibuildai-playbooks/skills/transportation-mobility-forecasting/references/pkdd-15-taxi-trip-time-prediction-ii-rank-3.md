# Method Sharing

Competition: pkdd-15-taxi-trip-time-prediction-ii
Rank: #3
Source: https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii/discussion/14988#83196

<p>Hi,</p>
<p>Here is&nbsp; our team's approach (BlueTaxi):</p>
<p>First we created our local training set by selecting the cut-off times the same as the five snapshots on the test set (the same week-date as well). We also see that 14th of august is the day before a big holiday in Portugal and 21th of December is the last Sunday before Christmas, we hypothesize that on that day people go to shopping and to the Church.</p>
<p>We created a bunch of features as follows:</p>
<p>1. Features from 10-nn, for every test trip we find 10 nearest neighbours w.r.t the Euclidean distance (DTW is an option but we see that it is not very efficient) , consider the durations of those trips as predictors.</p>
<p>2. Features from kernel regression, similar to 10-nn kernel regression was used to predict the duration, kernel regression is a smooth version of knn and these features showed very good results.</p>
<p>3. Some features from the partial trips: travelled distance, num_gps, last gps, average speed at different part of the trips, accelerations at different part of the trips </p>
<p>When matching a test trip with the training trips, we only consider to match the last 100, 200, 300, 400, 500, 1000 meters and the full trips as well. This is because the later part of the trip is more important in some cases. Our model show that the last 500 meters of the trip is very important.</p>
<p>We also considered contextual matching (match only trips with same taxi_id, the same week_date, the same call_id etc.) because we see different distributions of destination for these contexts. The kernel regression on taxi_id context gives the best results.</p>
<p>When modelling, don't predict the duration of the whole trip but predict additional travel time instead. Because the metric of evaluation is RMSLE so we should predict log of the additional travel time.</p>
<p>Outlier handling: we found that trips with missing values&nbsp; (identified at speed limits 160, 140, 100 Km/h) are more difficult to predict, we try to recover this information on the test set by looking at the gap between the cut-off timestamp and the start timestamp. Unfortunately, this information is not reliable so we decided to remove outliers&nbsp; with num_gps more than at 3.5 median of num_gps update.&nbsp;</p>
<p>We used different models, I used random forest and SVR while Bluebalam used extreme randomized trees, gradient boosting trees etc. Locally we see consistent results with CV&nbsp; (0.41). On the test set the result is quite different. But our approach is robust enough on both public and private test sets.</p>
<p>Cheers,</p>


<p>cheers,</p>
