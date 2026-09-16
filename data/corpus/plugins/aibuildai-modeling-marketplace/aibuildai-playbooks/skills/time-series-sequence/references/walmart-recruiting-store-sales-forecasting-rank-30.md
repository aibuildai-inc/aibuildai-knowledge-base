# Thank You and #2 rank model

Competition: walmart-recruiting-store-sales-forecasting
Rank: #30
Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023#44067

<p>In case anyone finds it useful, here is a little more detail to my strategy.&nbsp;Since I only submitted within in the last week, I had to find a way to validate my results locally without submitting and not use the leaderboard feedback.</p>

<p>My targets were the ratio of the previous year's sales to the current year. For example, this year's sales 20000, last years sales were 15000, so the target is 1.33.</p>

<p>By using the targets this way, it made sense to try and find the differences in the features from the current year compared to the last. The actual feature values helped a little as well.</p>

<p>Features were the difference from this year's values compared to last (CPI this year - CPI last year, Fuel Price this year - Fuel Price last year, etc), along with all of the features we were given on their own. The CPI and Fuel Price differences seemed to make the biggest impact out of these. I also added in the average department target across all stores, total sales for the individual stores, current week. I really wanted to find a way to use the markdown data, but it all failed. I left it in for my full model and it might have hurt the results based on what others have observed.&nbsp;</p>

<p>On top of those my best features were adding in the ratio of the sales from last year compared to the sales from last year the week before/after. For example, it's 2012 week 12, the features would be the (2011 week 12 sales) / (2011 week 11 sales) and (2011 week 12 sales) / (2011 week 13 sales). My reasoning for this working is that it could find whether there was a large/small drops in the weeks around the previous year's sales. Then when the target is a large/small value, it could be explained by a relatively large/small increase in these features from the year before.</p>

<p>Using this approach I was able to get good updates locally by training on dates&nbsp;before November 2011 and testing on dates after,&nbsp;and they&nbsp;corresponded well to the leaderboard results. I can see why using the leaderboard would have been more helpful, as this is leaving out a few months in the training set. My baseline to try to beat was just predicting the last year's sales since my final prediction was just (predicted ratio) * (last year's sales).&nbsp;I ended up using a single GBM trained on all of the data that had a record from the year before to be able to calculate all of the differences.&nbsp;</p>
