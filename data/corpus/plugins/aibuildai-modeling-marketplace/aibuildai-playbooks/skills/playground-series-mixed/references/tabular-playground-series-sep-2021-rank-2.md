# 2nd place solution - (A lot of) Stacking

Competition: tabular-playground-series-sep-2021
Rank: #2
Source: https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/275740

Hello everyone,

First of all, I would like to thank the Kaggle Team for hosting such great challenges and the community here for sharing lots of helpful knowledge.

I would also like to thank @mlanhenke, @realtimshady, @tunguz, @lucamassaron, @firefliesqn, and lots of others on the discussion board for sharing their approaches. Many ideas and models were based on their notebooks, and they were the reason I created such a diverse stacker.

Actually, I didn't expect the result of 2nd place, because I mostly used basic methods and techniques. The most extreme implementation was the number of the base models, reaching a number of 115 at the end. The general structure can be seen in the figure below with more details following.

[[tps-sep.png]](https://postimg.cc/gwRj9ykh)

## Preprocessing
- Various statistics for per instance (e.g., number of missing values, standard deviation, min, max, average, etc.).
- Filling all or some null values with mean, median, and mode depending on the variable's distribution.
- Scaling data with different scalers (e.g., standard, robust, etc.).
- Creating categorical features based on the initial features.
- Adding k-means, outlier, t-SNE, and UMAP features.

## Base-Models
- XGB, LGBM, and CatBoost were trained on CPU and/or GPU and optimized using Optuna.
- Histogram-based Gradient Boosting Classifier was also optimized with Optuna.
- TabNet was used with or without pre-training and achieved acceptable performance.
- Generalized Additive Models added more diversity to the final architecture.
- These models were trained with different parameters and features, creating a large level 1.

## Meta-Model
- Linear models worked best (L1, L2, Elastic Net regression).
- After experimentation, the Elastic Net seemed to have the best CV score, which was selected for the final submission.
- The first output of the Elastic Net model was fed into the Elastic Net model one more time to get the final submission.

Finally, I learned a lot through this competition. Probably, the most valuable thing I've learned is that it is important to do the basics carefully and experiment with different approaches. Moreover, implement a stable and working pipeline early on and trust it works throughout this process (CV was reliable in the final submission).

Again, thank you all here. I really enjoyed this competition.
