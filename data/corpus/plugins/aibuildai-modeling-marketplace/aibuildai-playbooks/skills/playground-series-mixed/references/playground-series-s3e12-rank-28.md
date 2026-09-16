# # 28 Solution | Problem with private LB score

Competition: playground-series-s3e12
Rank: #28
Source: https://www.kaggle.com/c/playground-series-s3e12/discussion/402347

## Hi Kaggle,

First and foremost, I would like to express my gratitude to all the participants who took part in this competition. I am thrilled to have secured a top 3% position in this playground series, especially considering that this is only my second time participating in such an event. I must acknowledge that my success is attributed to the valuable insights and knowledge-sharing that took place during the competition. I have learned a great deal from these discussions and contributions, and I am truly appreciative of the efforts made by fellow Data Scientists in enhancing our collective education.

I would like to give special thanks to these notebooks: 
* https://www.kaggle.com/code/tetsutani/ps3e12-eda-ensemble-baseline
* https://www.kaggle.com/code/mmellinger66/ps3e12-eda-xgb-cat-lgbm
* https://www.kaggle.com/code/johnycooly/random-forest-with-optuna
* https://www.kaggle.com/code/klyushnik/episode-12
* https://www.kaggle.com/code/sujithmandala/playground-s3e12-eda-logistic-regression

And also this discussion topic:
* https://www.kaggle.com/competitions/playground-series-s3e12/discussion/401188

## My Best Subbmited Solution 
I think, like many others, my best private score remained unsubmitted, but I will write about this later in this topic. Now I want to discuss the decision that led me to 28th place.

### Feature Engineering
* For the most part I ended up using feature engineering from this notebook: https://www.kaggle.com/code/tetsutani/ps3e12-eda-ensemble-baseline/notebook 
* I also added some binary features like `hydration status`, `osmo_status` and `calc_status`
* The part with outlier removal was also very important and it played a huge role in my understanding of public score 

### Models
* During competition I was trying differend combinations of models ensembles and in this solution I used weighted ensemble of optimized with optuna XGBoost, CatBoost, LGBM, GradientBoosting, Logistic Reggression, RandomForest, SVC and KNN.
* I also used 10 kfold splits to train models. While the dataset is very small 10 kfolds may seem like too much, but I have tested different values like 3, 5, 7, 8 and 12 and 10 seemed the most optimal for me.

## Problem I found with LB scoring

In total, I made 26 submissions for this competition, most of which were aimed at optimizing the model ensemble by trying different combinations of models. Whenever I attempted to change something with feature engineering, the score always fell (not including the outlier part).

As a result, most likely due to my inexperience in competitions, I reached a point where I could no longer achieve a higher score than the one I had already received. Although theoretically, it seemed to me that some solutions should perform better because I took more care to prevent overfitting and ensured that the model performed better on cross-validation. Despite my efforts, I could not surpass 140th place in the top public scores.

However, I noticed something strange when I started experimenting with the IQR method for outliers and tried different k values. I assumed that most participants are likely to overfit their models for the public leaderboard, even though it only accounts for 20% of all data. Therefore, I decided to choose the most optimal solutions for my final submission. It turned out that my assumptions were still not quite correct.

I had a solution that was similar to the one described above but used fewer models. I excluded KNN and GradientBoosting and used 0.3 and 0.7 quantiles instead of 0.25 and 0.75. This decision resulted in a modest score on the public leaderboard, but would have brought me to 16th place in the current top.



## Conclusion
As a result, although I have some assumptions, but I am still not completely sure what caused such a result in this competition. However, in any case, I'm happy with the result, I'm glad that I participated and learned a lot for myself, and thanks to everyone again for the wonderful experience and see you in the next competitions.
