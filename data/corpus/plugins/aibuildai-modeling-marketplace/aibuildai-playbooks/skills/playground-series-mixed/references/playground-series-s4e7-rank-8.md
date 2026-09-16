# 8th Place Solution

Competition: playground-series-s4e7
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s4e7/discussion/523486

For our final model, we used CatBoostClassifier with the following parameters:


```python
model = cb.CatBoostClassifier(
    iterations=30000,
    learning_rate=0.02,
    random_strength=0.1,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    leaf_estimation_method='Newton',
    random_state=1,
    subsample=0.9,
    bootstrap_type='Bernoulli',
    task_type='GPU'
)
```
We also used the [target reversals](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/520253) method on our submissions.

**GPU Setup:**

We utilized two GPUs for our training process:

NVIDIA GeForce RTX 4060 - My personal computer GPU, was used for the initial model training and testing phases, it was much slower but sometimes had higher scores.
NVIDIA A100 - Provided by Bar-Ilan University, this GPU enabled us to efficiently handle the large datasets and extensive iterations required for our model.

**Cross-Validation and Results**

We employed a 10-fold StratifiedKFold cross-validation approach, running 30,000 iterations for each fold. This method ensured that our model was robust and generalizable across different subsets of the data. The following are the best iterations and corresponding ROC-AUC scores for each fold:

Fold 1: Best Iteration = 19,186, ROC-AUC = 0.895919
Fold 2: Best Iteration = 19,431, ROC-AUC = 0.896413
Fold 3: Best Iteration = 17,659, ROC-AUC = 0.895431
Fold 4: Best Iteration = 15,656, ROC-AUC = 0.896021
Fold 5: Best Iteration = 16,038, ROC-AUC = 0.895603
Fold 6: Best Iteration = 18,125, ROC-AUC = 0.895498
Fold 7: Best Iteration = 14,044, ROC-AUC = 0.895487
Fold 8: Best Iteration = 14,509, ROC-AUC = 0.895475
Fold 9: Best Iteration = 15,585, ROC-AUC = 0.895458
Fold 10: Best Iteration = 15,268, ROC-AUC = 0.895620

The total training time was 15 hours. The mean ROC-AUC score across all folds was 0.895693.

We had a great time participating in the competition and look forward to many more.

Also a huge congrats to [Ravi Ramakrishnan](https://www.kaggle.com/ravi20076) and [Minato Namikaze](https://www.kaggle.com/arunklenin) on their [first places amazing winning approach](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/523404).
