# 9th place solution | NCAAM 2021

Competition: ncaam-march-mania-2021
Rank: #9
Source: https://www.kaggle.com/c/ncaam-march-mania-2021/discussion/230991

Thank you Kaggle for hosting this interesting competition. It was very fascinating to see the results of live matches. I became Kaggle Competitions Master with a solo gold 🥇 in this competition with 75th rank out of 159k total competitors in Kaggle competitions globally!

I have made my solution public [here](https://www.kaggle.com/prashantkikani/ncaam-2021-diverse-model-ensemble) in this notebook. 

Basically, an ensemble of multiple diverse models with features highly inspired by some of the great public notebooks. Most of my focus was on the modeling part.

# Modeling

So, my final solution contains an ensemble of `LGB`, `XGB`, `HistGradientBoostingClassifier`, `RandomForestClassifier` and `LogisticRegression`. I also tried models like `SVM` & `LinearRegression` but it decreased the CV, so I excluded those models from the final ensemble.

Basically, my goal was to reduce the bias of a model as much as possible. 
Ensemble weights are decided by the CV score.

# CV

I preferred `GroupKFold` with the `season` column as a group over other alternatives as it will treat all the season together instead of random shuffling.

And obviously, I only used data until 2015 in the CV & model building to avoid any leakage.

The score on 2015 to the latest year data will be treated as test. So, any improvement in that score highly likely to convert to the 2021 score.

Also, I created some `magic` features out of higher importance features from the LGB feature importance graph. Which may have some effect as well. I would recommend seeing the [notebook](https://www.kaggle.com/prashantkikani/ncaam-2021-diverse-model-ensemble) for more details.

Let me know if you have any kind of confusion/questions. Happy to answer!
