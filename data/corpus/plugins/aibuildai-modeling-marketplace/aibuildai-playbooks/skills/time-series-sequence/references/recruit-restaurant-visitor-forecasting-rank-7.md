# 7th place solution

Competition: recruit-restaurant-visitor-forecasting
Rank: #7
Source: https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/49259

As has been widely discussed, I had trouble with CV schemes, engineered features, and with how to use the reservation data.  In the end i built many models based on a variety of ways for treating both of these.  

For timed-based CV I used 6-week blocks starting at week 14 of the data.  So I trained on weeks 1-13, evaluated on weeks 14-19;  trained on weeks 1-20, evaluated on weeks 21-26, etc.  This gave me a fold that had a golden week .  Doing this meant I had sets of training data developed only on what preceded the evaluation fold.  But then I also used a random 5-fold CV for some models.

For the reservation data, I ended up using it "indirectly".  I calculated the standard deviation of the reservation counts and reservation visitors per store for things like each city &amp; weekday, or each genre and weekday, or each ward, etc.  I used these to calculate an "expected visitor" number for each store based on the store's mean/stdev visitors, but using the std of the reservation counts or reservation visitors.  So if the store was in city X and that city had a stdev of 1.5, I got a expected visitors for every store in city X using (store mean visitors)+1.5(store stdev visitors).  For a single expected visitors value I averaged all of the estimates for each store.  The reservation count (i.e. number of reservations for a day) ended up being a better estimator than the reservation visitors.

Engineered Features.  I didn't use very many.  The best ones were: Reservation-based expected visitors Rolling averages (I used a simple linear interpolation to fill in missing days).  I also used these for an LSTM that didn't work out so well. Store competitor counts.  So store counts with similar genres that were within 1 mile, or within "walking distance" (800 meters) I also set my "golden_week" flag to 1 during the Christmas week.

Target feature.  My weekday rolling averages, after adjusting for holidays, was an almost overpowering feature in LGBM and XGB so for a couple models I changed the target feature to be visitors-the rolling average.  That really mixed up feature importances, and performed a bit worse than the visitor-target, but good enough to include in my ensembling.

Ensemble.  I used a Neural Network (one hidden layer) to combine the various models.

Thanks for reading through all that!  That solution didn't fair so well.  Because I lost the last two lottery competitions by sticking with my CV, I decided I had better choose an "overfit the public LB" model just in case.  So I averaged the .479 SurpriseMe!  with my best single public LB model, that used a 5-fold CV, and built all features using all of the training data (future included).  And that model got me 7th place :-)
