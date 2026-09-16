# Share some insights about LightGBM vs. CatBoost in Zillow Prize-1

Competition: zillow-prize-1
Rank: #23
Source: https://www.kaggle.com/c/zillow-prize-1/discussion/47283

My final ranking benefited from the only_catboost kernel: https://www.kaggle.com/abdelwahedassklou/only-cat-boost-lb-0-0641-939. It looks obvious that with little feature engineering, plain CatBoost performs much better than plain LightGBM, Here is what I found out.

With plain LightGBM, using sale-month as a categorical feature, even In-Sample prediction errors (the diff between predicted pricing error and real pricing error) are biased, i.e. if we group errors by month, we will find within each group, the errors have non-zero median. But errors of total population is unbiased, which means LightGBM is not doing a very good job in training local-structures. With plain CatBoost, there is no bias within each month group. I think this is the major reason CatBoost is performing better.   

To look into why, from CatBoost tech-websites, I found this reference: 'Fighting biases with dynamic boosting' (arXiv: 1706.09516). It says, if we directly train with Gradient Boosting, the errors are actually biased (here bias is defined in a different way from what I mentioned above), the idea to solve the problem is to not repeatedly use the same samples along iterations. I did not have time to look into CatBoost source code, but I assume that since it appeared in reference, the idea should have been applied. Also, it is consistent with the observation that CatBoost trains much slower. And with Zillow Prize-1, this bias problem is amplified by the fact that the data is of high noise, so small contribution from bias adjustment could have made a huge difference in ranking.

This is all I find out, looking forward to anyone else contributing ideas like how this error-bias problem actually leads to the training-bias within months, and how this bias-fighting idea is actually implemented within CatBoost.

Now, if anyone interested in my solution, I would like to share some general ideas.

Before knowing about CatBoost, I solved the LightGBM bias problem by adding one more layer of modeling on target months (10, 11, 12) errors. which could be viewed as model stacking, not from a different model-family on the same data, but the same model family on different data. By doing this, I achieve manual bias adjustment within the group for sure, and hopefully to capture some more month-specific local structures. And of course, with smaller sample size, for this second layer training, a different param set and a smaller hand-picked feature set is used.

Then, when I was settled with my model, I looked around and saw this only_catboost kernel less than 24 hours before deadline. I did not have much time nor submission opportunities to play with it, so I used much of it as it is, even with less ensembles. But the original kernel has several critical flaws that needs to be fixed:

- 1, there is no effective prediction for 2017, which is easy to notice.
- 2, there is no need to mark 2017 tax-features NA for training, just need to make sure to use 2016 features for 2016 errors and 2017 for 2017.
- 3, there should be no sale-day feature: letting alone whether there is pattern at that dimension, we do not have that information when performing prediction.
- 4, for month and quarter features, the original kernel used them as year-month, and year-quarter, i.e. same month from 2016 and 2017 are used as different groups. This is critically bad, it would make 2016 prediction (public LB) good, but 2017 prediction very bad, as 2017 values of these features would not appear in training data at all. 

Either one above used alone would have made into first 100th, 0.0749786 for CatBoost and 0.0748932 for 2-step LightGBM (I believe with some more careful tweaks CatBoost could do better), so I think the key in this competition is the training bias.     

Besides all above, I used little feature engineering. Just 2 things,

- 1, census_and_block info is not numerical, it is actually a concatenation of 3 pieces of number-like categorical information. First four digits contains the same info as FIPS, next 6 census, and last 4 block.
- 2, including census and block, there are several high-cardinal categorical variables. For CatBoost I directly used them as categorical; for LightGBM, I delegated them as group_by_mean against a popular numerical variable, which is chosen according to feature importance for each categorical variable. I did not have a chance to try this on CatBoost.

I also tried brute-force feature engineering, but found it not useful. For each numerical vs. categorical feature pair, I did within group: group-mean, group-neutral, group-norm and abs value of above, then select by feature importance. It turns out to be overfitting to CV set, and not performing well on testing data.    

This is my first Kaggle competition, I really learned a lot, and I hope this sharing is beneficial for someone else. 

Looking forward to the top teams sharing about their insights and tricks, especially on feature engineering, which I did not do very well.
