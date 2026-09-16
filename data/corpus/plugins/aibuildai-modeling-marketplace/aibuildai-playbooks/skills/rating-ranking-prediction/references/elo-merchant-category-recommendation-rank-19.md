# 19th place memo

Competition: elo-merchant-category-recommendation
Rank: #19
Source: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82178

Thanks to ELO for hosting such an interesting competition, and congratulations to all the top teams.
I will write a brief memo.
# **Final Submission**
Blending some lgb with different features
Private : 3.600
Public  : 3.681
# **Best Single Model**
12 fold lgb with 300 features
Private : 3.605
Public  : 3.688
CV      : 3.634
・Best cv model has best private score.
・I didn't any post processing.
# **Useful**
Aggregate one-hot-encoded merchant_id.
I had to select some merchant_id because there was no sufficient RAM.

    train['outlier'] = (train['target'] &lt; -33).astype(int)
    hist = hist.drop_duplicates(['card_id','merchant_id'])
    hist = hist.merge(train)
    hist_outlier = hist.groupby('merchant_id')['outlier'].agg([len, sum, np.mean])

Now, you can see some merchant_id have large value.
So you can select them.
For exsample.
```
hist_outlier[(hist_outlier['len']&gt;500) &amp; (hist_outlier['mean']&gt;0.02)]
```

#**Didn't work**
-- Modiling with NN, RNN, xgboost, catboost, FFM
-- Directly target encoding
-- Feature decomposition(TrancatedSVD, LDA)
-- Feature selection with boruta
-- Stacking models by logistic regression, lightgbm

#**Addition**
 I found out later that it was effective to predict target for each transaction row.
1. Merge hist(or new) transaction data and train data.
2. Build lightgbm model to predict target of each row.
3. Aggregate predictions by card_id. 
```
df.groupby('card_id')['predictions'].agg([min, np.mean, np.median, max, sum, np.std]))
```
These features improve Private 0.003 but make worse Public 0.003.
So I could't use these features in final submission.
<br>
That's all. Thank you for your reading.
