# 1st place: One vs Rest + Multiclass Models

Competition: playground-series-s6e4
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s6e4/writeups/1st-place-one-vs-rest-approach

First of all, thanks to Kaggle for this competition. This month dataset and metric were non-standard, which made it more interesting! Also, a huge thanks to @mahoganybuttstrings, @utaazu, @yunsuxiaozi, @include4eto, @yekenot, @mikhailnaumov, @rohit8527kmr7518, @rawashishsin, @wguesdon, @lucasmoraes001, and @ravi20076 for sharing the solutions I used in final submission.

# Approach

I used OOF predictions from public notebooks with different FE for ensembling. If public notebook contained only 1 model, I trained other models on same FE in some cases. 
I also made one FE and a set of 6 models myself, where FE is still based on public notebooks.

Key observation I used was mentioned in [https://www.kaggle.com/competitions/playground-series-s6e4/discussion/690677](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/690677) - models almost never confuse Low and High classes. Therefore, I developed an approach with two binary classifiers. The first trained on Low vs Rest target on whole dataset. Since High class part is small, it is almost same as Low vs Medium.

In the second step, I trained Medium vs. High classifier. Only Medium and High classes were used for training, but OOF predictions were generated for the whole dataset. I considered two options: train the second binary classifier on true Medium and High labels or on predicted by the first model. In the final solution, I chose the latter. 

Conversion to class probabilities was done by formulas
```
P(Low) = P(Model_1)
P(Medium) = (1 - P(Model_1)) * (1 - P(Model_2))
P(High) = (1 - P(Model_1)) * P(Model_2)
```

This gave XGBoost and RealMLP models with a CV of 0.9805 each.

# Ensemble

I applied same scheme with 2 binary classifiers for probabilities blending. Blending was made with transform to logits + LogisticRegressionCV on 5 folds. I also ensured that all OOF predictions were obtained on 5 folds cv. Final cv of ensemble is 0.98155.
For final predictions, I used two threshold search algorithms—both are greedy selection based on OOF balanced accuracy.

# Models

61 oof predictions of models were used in final solution, 30 of them are from @wguesdon, 6 from my approach described above and the rest are from other public contributors. In a few cases, I reran public notebooks with other models.
In total, there are 17 XGBoost, 11 LightGBM, 10 CatBoost, 9 RealMLP, 3 TabM, 2 LogisticRegression and a few others. 

Details can be found in the attached ensemble notebook, which contains both ensembling and training mode for my models.
I have also attached one of TabM models training, based on [https://www.kaggle.com/code/rawashishsin/s6e4-highest-score-xgboost-cv-0-98109](https://www.kaggle.com/code/rawashishsin/s6e4-highest-score-xgboost-cv-0-98109) which was not included in best scoring submission.

# Conclusion

There are many multiclass and multilabel problems where one-vs-rest is not optimal.
Here the dataset is specific because models almost never confuse Low and High classes. This made one-vs-rest approach with 2 binary classifiers possible and allowed to focus first on ranking and then on thresholds calibration for balanced accuracy.
