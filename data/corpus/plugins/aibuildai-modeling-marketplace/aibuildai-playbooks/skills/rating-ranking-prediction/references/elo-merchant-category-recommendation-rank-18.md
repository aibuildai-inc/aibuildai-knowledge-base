# 18th place story (pocket's side)

Competition: elo-merchant-category-recommendation
Rank: #18
Source: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82107

My journey
------------------------------------------
So this competition has been a long long journey for me.  
It felt way longer than it actually was, which is just 40 days. 40days ago, I wanted to take a break from another competition, and decided to do a "40days challenge".

I started off by aggregating a lot of the time-series, but it was hard to even beat the kernel. The data seemed to have so little signal, almost everything that I did didn't work.  
After a week or two, I somehow managed to score 3.636-3.630 locally, which happened to be around 3.608-3.605 private. I was running out of ideas, so I decided to team merge.

After merging with my team, we merged our features and re-run our models. Our best single model improved about +0.002 at that time.
We searched for a good way to ensemble our models, but couldn't find a good way, especially with all the post-process that was going on.  
Just one week before the deadline, @marcuslin finally found that a simple ridge including non-outlier model + binary-prediction model works really well. Our score jumped to the 3.66x range in public, which was 3.600 range in private.

For our final submission, we were torn between 3subs. One with heavy post-process, one with conservative post-process, and one without post-process.  
We were pessimistic about our final position, so we took the risk and selected the two with post-process. This was actually a mistake, but still good enough for 18th place.  
Had we selected our no-post-process sub, we would have placed around 10th place, but that is just another story.

I really feel lucky/happy about the final result, since in the heat of the competition, I could neither trust the CV nor could I trust the LB, which means I was flying blind.

Since some people may have interest, I will describe the technical side of things, my features and my models below.

------------------------------------------
Features
------------------------------------------
**Used features**  
1)Basic features: very good  
Aggregations like the kernel. Not much to say.

2)Time features: includes magic features  
Features about When there was a purchase.
The kernel lack "the last day of transaction", which is magically good.

3)Conditional features: good  
The aggregation when (authorized_flag == 1, 0), (city_id == -1)

4)Recent features: good  
The aggregation when (month_lag &gt;= -2, or 0)

5)Prediction of time: probably good  
Predict the magic_feature:(last day of month_lag2 - last purchase_day) from only the historical_transaction features.
Then do (predicted_days - actual days), and drop the predicted days.

6)Time features: maybe good?  
https://www.kaggle.com/denzo123/a-closer-look-at-date-variables
Hour and DayOfWeek seems to be very important (but hard to improve the cv-score).
I binned "hour" to [6-12, 12-17, 17-21, 21-05], which seems to add very small value(maybe).
By the way, 00:00:00 has a very high target value (probably because it is periodic purchases), but I couldn't improve the CV score with a flag, maybe because there are too few occurrence...

7)Target-encoding features: maybe good? destablized the score...  
OOF-target encoding of categories(merchant_id, subsector_id).

8)(+, -, /, *) of two features: maybe stablizes the score? not sure...

------------------------------------------------------
**Unusued features**  
I had tons of ideas which didn't work. Here are some of the interesting ones

LDA features: thrown away  
the LDA topic of each categories (like in TalkingData competition).
I guess categories have no information!

Sequential authorized_flag="N": thrown away  
Sometimes, there are many authorized_flag="N", in a sequence.

Periodic buys: thrown away  
Sometimes, there are periodic purchases. Like, always once a month with the same price, at the same merchant.

Encode features: thrown away  
Encode the categories by mean_purchase_amount.
Take the difference (purchase_amount - mean_purchase_amount).

------------------------------------------
Models
------------------------------------------
**Models/post-process**   
In general, stacking with simpler algorithms worked better, so Ridge/Linear regression seemed better than LGBM.
My teammates were very successful with stacking, so I will leave it for them to write.

------------------------------------------
**Models which didn't work/weren't used**  
Time-series prediction: didn't work  
Like in the HomeCredit competition, predict the target based on the raw time-series/monthly time-series. Neither worked.

Isotonic regression: didn't work  
I probably did this wrong, but I couldn't make it work

Random stacking: didn't use  
For stacking, I stacked upon automatically-generated-random-models by randomly selecting a subset of my features. We did not select the stack, since the publicLB score was bad, but the CV and privateLB score(3.660) is actually better than our final submission.
Need some more experiments to confirm.

Pseudo-labeling: didn't use  
Pseudo-labeling with our best stack seems to improve our base model on the CV and privateLB. But it did bad on the publicLB, so we didn't use it.
PL probably worked both on LB/CV earlier with NN, but we didn't have enough time to squeeze it into our final stack.

------------------------------------------
Conclusion
------------------------------------------
I really want to thank my teammates for doing a fantastic job in this competition.  
Without my team, I would not have been in this place.  
I also want to thank and congratulate @philculliton for his first competition without a leak!
I hope you won't ban more than 221 people so that I remain in gold :)

And.. I guess that is it.
Thank you all for participating in this competition, and good luck ;)
