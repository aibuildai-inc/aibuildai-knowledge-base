# 1st Place Solution | A Diverse Ensemble

Competition: playground-series-s3e15
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e15/discussion/414048

*This has been a great place to learn from members of the community and here is my small contribution in return.*

What a pleasant surprise! I have only started competing on Kaggle recently. I was hoping to jump up some places but not all the way to the top. As @iqbalsyahakbar said, the shake-up was massive.

# Key takeaways

- Ensemble with models that have diverse internal workings like tree-based methods, NNs, linear models, and nearest-neighbors, even if they have **worse** individual performance.
- Use domain knowledge to manipulate the data.
- Use the original data for training but not validation.
- Trust in your CV. I tried to look away from the public leaderboard scores and focus on improving the local CV score. I was hitting a limit and couldn't improve further. I knew either I was missing a trick (which I would be curious to learn about after the competition ended) or that ~0.072 was actually the limit of the performance for this dataset. A note on that: I had a submission with a better private LB score but didn't pick it as it had a worse CV score.


# Things that worked
- **Using domain knowledge for imputation and range clipping:** @shalfey [made an interesting remark about this towards the end of the competition](https://www.kaggle.com/competitions/playground-series-s3e15/discussion/413293). I realized that if the "author" column can be appropriately imputed, we can clip values of other features and reduce the noise in the dataset based on the "author" value. Also, originally posted by @arunklenin in a comment thread.

- **Categorical encoding** for "author" and "geometry" column.

- **Iterative imputation**: perform missing value filling iteratively using trees. Thanks to @arunklenin for sharing this [here](https://www.kaggle.com/code/arunklenin/ps3e15-iterative-catboost-imputer-ensemble#4.1-Impute-Categorical-features). I am curious how you tuned those parameters because they worked really well.
- **Tune model parameters** with optuna.
- **Diverse models** for ensembling. Tree-based and NNs among others. Even though some of these had worse performance, the way they work added diversity to the ensemble as pointed many times by @ambrosm in previous competitions. CV for the single best model was 0.0730 but ensembling brought it down to 0.726.

- **Proper cross-validation** 10-fold CV that lead me to a CV score of `Ensemble RMSE score 0.07265 ± 0.00202`. 

# Things that didn't work
- Clipping target variable values based on domain knowledge.
- One hot encoding for "author" and "geometry" columns didn't really help. It also increased the fit time as there were more features available.
- Adding PCA features didn't help.
- I experimented with an auto-encoder but that didn't seem to add much. Probably didn't cross-validate this one properly.
- I also experimented with clipping all features values to +-3 standard deviations to reduce outlier values. This gave a better score in the CV but seemed too optimistic at around ~0.069. Indeed the leaderboard for that submission was worse.
- Out-of-the-box imputers were worse. I believe the tree based worked best as there was almost an "if-else" pattern depending on the values of other features like "author" and "geometry" among others.
 


Thanks to everyone who participated and shared interesting ideas. Until next time!
