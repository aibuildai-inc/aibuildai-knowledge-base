# 8th Place Solution

Competition: jpx-tokyo-stock-exchange-prediction
Rank: #7
Source: https://www.kaggle.com/c/jpx-tokyo-stock-exchange-prediction/discussion/359227

First of all thanks to JPX for hosting such an interesting competition and thanks to all contributors that shared their ideas and code. It helped a lot to understand what the goal was and how to approach it. This was our ( @MrChrissie ´s & my ) first competition in the area of finance and we gained a huge amount of knowledge through other contributions. 
This post is inspired by @flat831 who posted his team solution earlier.

**Approach and Model:**
After a lot of trial and error, we came up with a relatively simple idea. 
We divided the stocks by the 33SectorName into groups and trained a separate LGBM regressor model for each group. Our entire model then consisted of one LGBM model for each sector. This idea came under the assumption that stocks of the same sector tend to correlate. The prediction target of our model was the change ratio aka. the target column. For the submission, we ranked the predicted change ratio.

Since we ran out of time in the end, a single LGBM model was tuned to all stocks using Optuna and the resulting hyperparameter values were then used for our sub-models. 
We only tuned the following parameters:
- num_leaves
- max_depth
- learning_rate
- n_estimators

For the training,  we used available features from the stock_price and stock_list files.
