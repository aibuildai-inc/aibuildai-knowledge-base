# 9th place solution + Pseudo label + IterativeImputer + Multi AutoGluon

Competition: playground-series-s3e15
Rank: #9
Source: https://www.kaggle.com/c/playground-series-s3e15/discussion/413808

Only had a few hours with this competition as I started the same day but nevertheless everything was possible and glad to see the 9th place 😊

The final solution is a mix of some ideas that I thought was possible before deadline:

1.	**Extend the training data with pseudo labeling:** Pseudo label the missing target with an ensemble of the top 5 public solutions .

2.	**Feature Engineering with Sklearn IterativeImputer with a DecisionTreeRegressor:** As the training data had many missing values both in target and in the features Sklearn IterativeImputer with a DecisionTreeRegressor could be a good alternative and later ensembled to the other standard handling of missing values.

3.	**Use AutoML frameworks to speed things up** and is a great option specially in regression problems as they comes with lots of different models, which is good to have in the final ensemble.

With the above ideas I trained a mix of them in total of 5 solutions with Auto Gluon framework, e.g. one plain untouched training pipeline and one with Feature Engineering with many iterations of Sklearn IterativeImputer + DecisionTreeRegressor, 2 stage pseudo labeling (first on public then from the trained AG models) and also with the AG FTTransformer to the models.

The 5 trained solutions then where weighted ensembled based on the result.

That’s it! 😊
