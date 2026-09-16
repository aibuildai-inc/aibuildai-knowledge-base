# Public 12 Place Short Solution

Competition: optiver-realized-volatility-prediction
Rank: #18
Source: https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/275169

Thanks to Kaggle, Optiver for hosting this competition, and thanks to a lot of public sharing from which I learned a lot. Finally thanks to my teammates for their efforts during the competition period.

#### **Validation Strategy**
Since the public\private LB represent the future market, the best way is to do a time-based validation split. However, since it is declared that we don't have an actual sequence relationship between different time_ids,  the best way to do validation is to do time_id based group k-fold validation.

The rule of thumb for validation becomes: 
1. Check every set of changes
2. if the cv holds or improves, we test on LB. 
3. If LB is improving, we keep those features.

The idea is that what features\models we created\tuned should generalize to the different buckets of time at least. Also, we treat LB as an extra validation fold that should represent the private test set better.

#### **Modeling**

##### **[Ideal NN Design]**
- Use RNN to extract book\trade features for each stock_id at each time_id
- Use Transformer to capture inter-stock relationship

[NN Design]

##### **[Evolution 1]**
Issue: The training was too slow
Solution: Replace RNN parts with handmade features
	- training is much faster
	- almost no difference in performance
	
**Performance: cv: 0.211, lb: ~0.200**


##### **[Evolution 2]**
Issue: As hint by the top team, we did not consider the relationship between time_ids.
Solution: Use Nearest Neighbors to find top-N closest time_ids, and average the features for each stock from the time_id neighbors as new features.

**Performance: cv: ~0.198, lb: ~0.189**


##### **[Blending]**
Transformer + Tabnet + NN with different feature sets.
