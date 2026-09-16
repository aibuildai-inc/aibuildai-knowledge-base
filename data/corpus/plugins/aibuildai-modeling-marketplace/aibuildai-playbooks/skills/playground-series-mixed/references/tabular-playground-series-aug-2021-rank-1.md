# 1st place, pseudocode

Competition: tabular-playground-series-aug-2021
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-aug-2021/discussion/270051

Hi everyone,

First of all, kaggle community is really fun place for people as me. I'm fairly new here and I'm really glad that I decided to "just a try" and participate in some of kaggle competitions. TPS is something that I like particularly.

About TPS for August:

I had that idea, which is in core very simple, but again maybe not.
What if we have good enough pseudo labels for test dataset?
What if we train model on that combination?
You maybe see what I'm thinking 😊
 
```
# test_df, train_df, train_y are self explanatory

test_y = pseudolabels
learning_rate = 0.2

lgbm_reg_1 = lgb.LGBMRegressor()
lgbm_reg_1.fit(test_df, test_y)

new_y = train_y - lgbm_reg_1.predict(train_df)
# now we have labels with positive and negative values ('loss' is not good word anymore, 'error' is better)

lgbm_reg_2 = lgb.LGBMRegressor()
lgbm_reg_2.fit(train_df, new_y)

error_prediction = lgbm_reg_2.predict(test_df)
test_y = test_y + (error_prediction * learning_rate)
```
We can do this several times while our score is getting better and last test_y use for submission.csv

This is some sort of boosting algorithm. I tried to find if someone use something similar, unsuccessfully.

I made new initial [notebook](https://www.kaggle.com/ivankontic/003-final-my-boost-1st-place?scriptVersionId=73878373), my final notebook is really mess and I'm embarrassed to make it public 😊

Thanks for reading
Ivan
