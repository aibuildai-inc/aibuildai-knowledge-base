# 26th place - full solution in R

Competition: ga-customer-revenue-prediction
Rank: #26
Source: https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/81639

[This is a repost of a comment I made to another thread.](https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/73260)

Here's my [source code][1] (and the [data prep step][2]). 

This was my first competition, and it took some time to get the basics right. Luckily I only joined in the v2 phase. As others have mentioned, the data was very sparse, and I found it very hard to beat the baseline (predict the historic mean) by any margin. However, I was able to get reasonably consistent results with a suitable **validation strategy**:

    cut_offs = ymd("2017-01-15") %m+% months(0:17)  # earliest we can start and have 5.5 months data (to make a consistent recent visitor pool)
    
    train_cut_offs = cut_offs[cut_offs &lt; max(cut_offs) %m-% months(2)] 
    
    valid_cut_offs = train_cut_offs %m+% months(2)

    train_sets = lapply(train_cut_offs, function(t) which(fm$cut_off %within% interval(t %m-% months(12), t))) 
    
    valid_sets = lapply(valid_cut_offs, function(t) which(fm$cut_off == t))

In other words, I had a cut off and new forecast every month, 18 times. I then built 18 sets of features using an expanding window, from the data’s origin to the cut off. This worked better for me than a rolling window. More on why later. The cut off is the analog to the 15th October 2018: the point at which the training data stops. I did not use the demo account to infill 16th-31st October, as I was trying to infill *all* the missing data, but of course @alijs execution above is much smarter.

For cross-validation, I would train each model with the feature set from the cut off month and the 11 months before, so I had a rolling window of 12 feature sets. (Note, each feature set is built from an expanding window, so they overlap.) This was an important step to **augment the data**. There’s so few conversions of recent visitors in any 2 month period, it’s hard to train a model of any complexity using a single cut off. I chose 12 months somewhat arbitrarily, as 5 or 8 months worked well too, arguing to myself it would smooth out any seasonality.

The **metric** I used for cross-validation was R². It scales the MSE vs the baseline: predict the historic mean. I felt this was helpful, as the MSE moved quite a bit over time, and this helped me understand why: the mean moved a lot, presumably due to promotional activity (see @MichaelP's comments). As you can see from this chart, the model was way off last Spring, so my entry relies on this promotional activity (or lack thereof) not happening again this Winter!

![R² by target interval end dates (cut off plus 3.5 months)][3]

I am willing to bet the winning entry will incorporate both short term knowledge of promotional activity, and long term general behaviour of visitors, especially repeat purchasers.

In terms of **features**, I took inspiration from *[Buy ‘Til You Die][4]* models. They have [sufficient statistics][5] of: the number of purchases, and the time since the first and the last purchase. This is why I used an expanding window: these models rely on having the entire history of transactions to determine whether the customer is still alive: they may not have transacted the last 6 months, but this could be just because they tend to purchase infrequently, say once every 8 months on average. These sufficient statistics were the main time features I used, both for visits and transactions (ie all visits, and those visits with positive `transactionRevenue`). I also added average pageviews per visit and average transaction revenue (ie sum of `transactionRevenue` over count of transactions).

I **mean-encoded** a few of the more useful categorical variables, but *not* using target encoding. The target was too sparse for this, so I encoded the historical conversion rate instead (total transactions / total visits for the specific value of the categorical).

During the competition, I thought I had found an important **data leak**. In the v2 train data, the organisers had removed the `source` column for a large number of visits. Comparing this to the v1 data, I could see these were consistently identifying internal websites for Google employees. I reasoned that Google employees were much more likely to transact, which is why they had redacted them in this way. So I added the redacted `source` back to the data set as a feature, for these visits, as well as an indicator. In practice, they  did not improve the model's performance once I included all my time features, described above, so I removed them.

To speed up my development cycle, I **rebalanced** the “classes”. I took the whole history and removed 90% of the non-transacting visitors. For the final submission I used the full set, of course. I say "classes" as I treated the comp as a regression, not a classification as others have mentioned. My reasoning was the positive cases were very sparse, so building two models was pushing it!

I used **Catboost** as it outperformed LightGBM, despite the fact that I had mean-encoded all the categoricals before handing the data to the model (Catboost specialises in mean-encoding categoricals, amongst other things). I didn’t try Xgboost. I didn’t ensemble or stack — I haven’t had time to learn this skill yet!

I also totally failed to pick 2 submissions for the private leaderboard! I thought I had to do that after stage 1 finished, but I now realise it’s something you need to do before, to avoid gaming. Anyway, luckily I’d only ever made 3 submissions, and the second highest scoring one (on the stage 1 leaderboard) is the one I wanted to submit, and it will go through automatically. As **Mr Miyagi** would say, [“You beginner luck!”][6]


  [1]: https://www.kaggle.com/burritodan/gstore-2-final
  [2]: https://www.kaggle.com/burritodan/gstore-1-flatten
  [3]: http://i67.tinypic.com/28tg6x1.png
  [4]: http://www.brucehardie.com/papers.html
  [5]: https://en.wikipedia.org/wiki/Sufficient_statistic
  [6]: https://www.youtube.com/watch?v=J1gAHil89Z4
