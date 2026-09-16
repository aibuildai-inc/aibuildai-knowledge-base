# 14th Place Solution: hillclimbers w/ negative weights

Competition: playground-series-s3e15
Rank: #14
Source: https://www.kaggle.com/c/playground-series-s3e15/discussion/413749

Hi everyone, I didn't have as much time to work on this competition but I still wanted to share my solution & thoughts. These competitions are always fun to be a part of and I really appreciate everyone who contributed and participated.

Here are the people that created strong models that I would like to give thanks (please upvote their work):
- @arunklenin | notebook: [🌋 PS3E15 | Iterative CatBoost Imputer|Ensemble](https://www.kaggle.com/code/arunklenin/ps3e15-iterative-catboost-imputer-ensemble)
- @tetsutani | notebook: [PS3E15 EDA| Ensemble and Stacking baseline](https://www.kaggle.com/code/tetsutani/ps3e15-eda-ensemble-and-stacking-baseline)
- @onurkoc83 | notebook: [KNN İmputer (LB score :0.075094)](https://www.kaggle.com/code/onurkoc83/knn-mputer-lb-score-0-075094)
- @iqbalsyahakbar | notebook: [PS3E15 | EDA, Imputing, Ensemble for Beginners](https://www.kaggle.com/code/iqbalsyahakbar/ps3e15-eda-imputing-ensemble-for-beginners)

I ended using my project [hillclimbers](https://github.com/Matt-OP/hillclimbers) (again) to create the final ensemble. I won't go through all the details of hillclimbers, but here is a [link to my 4th Place Solution](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/410639) in the previous Playground Series episode where I also used hillclimbers and you can find more in-depth explanations. Here is a visual presentation of the final submission:



A couple things to note:
- My model (the first hillclimbers model) used @arunklenin's preprocessing & feature engineering. I manually tuned the hyper-parameters for every model.
- I noticed there was heavy reliance on the following 3 models in a lot of the public notebooks: XGBoost, CatBoost, LGBM. This is not specific to this particular competition, but more so to the Playground Series as a whole. The goal of building my model was an attempt to **diversify** the predictions even though I was still using several gradient boosting models.
- The model I personally created ended up with a better Private LB score: **0.072698** than the final submission **0.072741** despite having a lower CV score which took me by surprise. I didn't end up choosing it as one of my two submissions.

Here is the weights used for the hillclimbing models:





**Thanks again, I am open to any and all feedback!**
