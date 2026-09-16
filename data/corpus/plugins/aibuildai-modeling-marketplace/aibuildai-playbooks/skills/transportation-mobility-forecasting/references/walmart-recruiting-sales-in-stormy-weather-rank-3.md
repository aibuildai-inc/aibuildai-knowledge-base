# Final Result

Competition: walmart-recruiting-sales-in-stormy-weather
Rank: #3
Source: https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/discussion/14358#80198

<p>Nice job all, I love the variety of approaches</p>
<p><strong>This is the general solution for #3 finish</strong></p>
<p>As many noted, we can zero out all the stores/items in the training data with zero item counts. This eliminates a ton of training and test data.</p>
<p><span style="line-height: 1.4"></span></p>
<ul>
<li><span style="line-height: 1.4">Store_nbr and item_nbr get their own one_hot encoding</span></li>
</ul>
<p><span style="line-height: 1.4"></span></p>
<ul>
<li><span style="line-height: 1.4">Discard CodeSum from weather.</span></li>
</ul>
<p><span style="line-height: 1.4"></span></p>
<p>Clean up and include all the relevant weather stats that fell under the the 'common sense' rule. Each training point got the weather data for that day, and the average of the 7 days prior and after, <strong>for each weather variable</strong>&nbsp;within reason &nbsp; (i.e &nbsp; &nbsp; snowfall_7_prior_mean, &nbsp; snowfall_that_day, snowfall_7_after_mean ) &nbsp;Plenty of NAs show up but less if you allow the mean function to remove NA's.</p>
<p>Then we of course include the data for dates. &nbsp;This deserves a little expanding upon. &nbsp; I was using trees and if you code everything one:hot as a categorical, you get some pretty rapid combinatorial&nbsp; explosion in your trees which can over-fit. &nbsp;As this competition seemed like under-fitting might be a bit better policy. &nbsp;So <strong>everything in the date category was coded as numeric.</strong> &nbsp;Boosting can still tease out information from the days of the week if you encode them as 1:7 or months 1:12 or days 1:365 or days 1:31 (ironically it picked up the new years day drop in sales without me telling it. &nbsp;Day_1:31 caught the monthly decline in sales without me explicitly looking for it. &nbsp;I was expecting 1st and 15th bumps but those weren't as obvious) &nbsp;It also allows for the computer to blend months and days together that behave similarly, <strong>if they are near each other numerically</strong>, so there is <strong>some</strong> order in the madness. In my view you get most of the benefit and tons of speed up. &nbsp;</p>
<ul>
<li><span style="line-height: 1.4">One feature that helped immensely ( Shize Su mentioned it) was as.numeric(Date). Its just a numeric vector from day 1 to day N.</span></li>
</ul>
<p style="padding-left: 30px">As.numeric(Date) catches all kinds of weird behavior.</p>
<p style="padding-left: 30px">i.e. some stores just abruptly stopped or started selling certain items at certain dates for whatever reason or had increasing or decreasing sales trends. &nbsp; It also kind of picks up weird stuff from other stores/items that are near that date numerically. &nbsp;And because its just a numeric column, you can throw it in for nearly free.</p>
<p>So the Date data only was 5 columns or so&nbsp;</p>
<p>( day_numeric_1_7,&nbsp;month_numeric, date_1_365 numeric, &nbsp;date_1_1035 numeric, day_1_31_numeric )</p>
<p>&nbsp;weather was about 40 columns if I recall, plus store and item</p>
<p>To algorithm land:</p>
<p>Since I am addicted to xgboost - I used it.</p>

<p>FOR i = 1:1000</p>
<p>%do%</p>
<p>X = a full bag of training data (i.e sample<strong> with replacement</strong> nrows from training data), y = <strong>log1p(y)</strong></p>
<p>xgb.train( params are random <strong>yes random</strong> as is reasonable, new params for each i of loop)</p>

<p><code>what it looks like<br> <br> param = list( <br> booster = 'gbtree',<br> objective = 'reg:linear',<br> eval_metric = 'rmse',<br> max.depth = sample(3:10,1), <br> eta = runif(1,.005,.5),<br> gamma = runif(1,0,10),<br> min_child_weight = runif(0,10),<br> subsample = runif(1,.5,.8),<br> colsample_bytree = runif(1,.7,1),<br> nrounds = 3000<br> )</code></p>
<p>train with early stopping,nice new feature on xgboost&nbsp;</p>
<p>predict on the out-of-bag data not used in training, save results and indexes</p>
<p>predict on test set, save</p>
<p>save params, save model, SAVE everything!!</p>
<p>save indexes, models , SAVE&nbsp;</p>
<p>Part one surprisingly only takes 4-5 hours on a quadcore and only uses modest RAM. &nbsp;we can rerun this procedure easily again and sample from good scoring hyperparams, or check cv_scores. Computers love busy work,so I give it to them.</p>
<p>At this point we have 1000 models, predictions, cv_scores, params and a leaderboard score of ~~.098 (simple linear average of 500 best models) . Because we have saved the indexes of the validation sets we can reconstruct out-of-fold behavior and train a lev2 model.</p>
<p><strong>lev2</strong></p>
<p>Used xgboost new feature 'boost from prediction'. &nbsp;I only want to nudge my predictions a bit. The difference between .098 and .094 is sooooo small. I noticed that december estimated low on weekdays, july high on weekends.,etc,etc There were discernible patterns in the error profile. So I train lev2 model</p>
<ul>
<li><span style="line-height: 1.4">X = &nbsp; [ lev1_prediction_out_of_fold, day_categorical, month_categorical ] &nbsp;</span></li>
<li><span style="line-height: 1.4">y = log1p(y) &nbsp;</span></li>
</ul>
<p>use same loop as in part one, but smaller learning rate, only 100 or so times you get to watch the RMSE decline.. sooo nice</p>
<p>predict with this model on test predictions from lev1. &nbsp;~ .094 &nbsp;</p>
<p>What this does is takes your original predictions and tweaks them with new data you provide. I decide that the day_categorical and month_categorical helped a bit. I think I could have gone lower but ran out of time. It is actually using the size of the original lev1 prediction as new numeric feature. i.e small values in July on Wed should be higher, large values on weekends in December should be lower... &nbsp;</p>
<p>Even though getting this set up was a headache, its really pretty simple and quick.&nbsp;</p>
<p>A couple points. &nbsp;xgboost is fast but has strange default missing values [0], if you don't override that it will learn where to put 0 values on its own. This is fine but can lead to unexpected behavior.</p>
