# Rank-3 Public Rank-5 Private Approach

Competition: playground-series-s5e8
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/rank-3-public-rank-5-private-approach

Hello all,

Greetings!

I am extremely happy to present my solution and writeup for the rank -5 at the Playground Season 5-Episode-8 competition. I wish to extend sincere thanks to Kaggle for the episode and my fellow participants for a rich and diverse sharing in public forums. My final solution for the competition is a concoction of the ideas presented in the below kernels, apart from my own inputs - 

- https://www.kaggle.com/code/mikhailnaumov/no-blending-bank-classification-xgb-lgbm-cat-ydf
- https://www.kaggle.com/code/itasps/0-97705-cv-0-97604-pg-s5e8-single-xgb
- https://www.kaggle.com/code/mahoganybuttstrings/pg-s5e8-single-xgb-cv-0-975782-lb-0-97681
- https://www.kaggle.com/code/bizen250/bank-term-deposit-single-catboost
- https://www.kaggle.com/code/yekenot/ps-s5-e8-deeptables-nn
- https://www.kaggle.com/code/molozhenko/playground-s5e8-advanced-lightgbm
- https://www.kaggle.com/code/cdeotte/xgboost-using-original-data-cv-0-976
- https://www.kaggle.com/code/cdeotte/train-more-xgb-nn-lb-0-9774

Kindly peruse my approach for the solution below-

# Feature engineering 

- I created features based on a simple prompt to ChatGPT and used them - none of them were effective to a massive extent!
- I used 2-gram, 3-gram and 4-gram features and used them as inputs with original as rows and columns. I avoided 4-gram features for original as columns (due to memory constraints)
- As illustrated in many public kernels, I used all category columns as **strings** and used the tree based internal encoders to treat them during the cv-regime 
- I used mean and count encoders inside the fold as primary features. I also used count encoder outside the fold, using the `train + test set` as a combined entity. I observed that this was a rare example where the data leakage did not harm the model. This can be illustrated as 

```python
(
    pl.concat( [ train.select(pl.col(col, "id")), test.select(pl.col(col, "id"))] , how = "vertical_relaxed" ).
    group_by(pl.col(col)).
    agg("id").count()
)
```
- I used polars for feature engineering and stored the resultant datasets with partitions using parquet files and hive. This helped me retrieve information easily and quickly. Each partition had 50,000 rows. I also reduced memory wherever applicable using down-casting techniques.

# CV-scheme 
- I used Stratified 5-fold cv scheme using state = 42
- All my model runs adhered to the same cv-scheme all throughout 
- I did not indulge in any full-refits and random state adjustments in this pipeline and rather resorted to models with varied features instead

# Offline model training - single models 
- I resorted to boosted trees, MLP, TABM models using a gamut of features, using the original as rows /columns throughout
- I also created pseudo labels for these models and trained a 2-stage pipeline with/ without pseudo labels 
- I also trained 1 Autogluon model on the raw and encoded features and included the results **only for pseudo labels** - these models were not used for any further steps

# Offline model training - stage 2 ensemble 
- I resorted to Autogluon for this stage. I trained 7 autogluon models with 16 hour runtime, using varied feature and models as a stacking ensemble 
- I tried other AutoML tools like H20.ai here, but Autogluon fared far better and I used only Autogluon for the final stage ensemble
- This was a very effective element in the ensemble and provided a massive CV boost though 

# Offline model training - stage 3 ensemble 
- I resorted to a hill-climb using several candidate models as inputs. I designed 10-hill climbs using negative and positive weights as options
- I also used 5 logistic regressions using several model candidates from single models and stage-2 ensembles
- My final submission is a 50-50 blend from the hill-climb and logistic regression results
- My alternative submission is the result of the stage 2 Autogluon with 320 model candidates. My CV score maximized here and I used this candidate as a final submission

# CV results 

## Without pseudo-labels

| Model algorithm | CV score range  |
| --- | --- |
| XGB | 0.97161 - 0.97645  |
| LGBM gbdt | 0.9712 - 0.9762 |
| LGBM goss  | 0.9713 - 0.97625 | 
| Catboost | 0.9707 - 0.9751 |
|  MLP | 0.9728 - 0.9741 |
|  TABM | 0.9731 - 0.9758 |

## With pseudo-labels

| Model algorithm | CV score range  |
| --- | --- |
| XGB | 0.9718 - 0.97652  |
| LGBM gbdt | 0.97125 - 0.9763 |
| LGBM goss  | 0.9714 - 0.9763 | 
| Catboost | 0.9708 - 0.97545 |

## Ensemble 

| Model algorithm | CV score range  |
| --- | --- |
| Autogluon | 0.97656 - 0.97745  |
| Hill-climbers| 0.97676 - 0.97746  |
| Logistic Regression| 0.97635 - 0.9773  |

# GPU stack

| Model algorithm | GPU  |
| --- | --- |
| Autogluon | L4 Colab  |
| Hill-climbers| L4 Colab  |
| XGB | A6000 Ada, A6000, A5000, 4090 |
| LGBM| 4090  |
| Catboost| L4, A100|
| MLP| T4 X 2|
| TABM| T4 X 2|

# Final comments

Wishing you the best for your future competitions personal and professional endeavors! See you around in the upcoming playground episodes!

Regards, 
Ravi Ramakrishnan
