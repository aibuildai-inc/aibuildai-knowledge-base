# #5 | Beginner's luck

Competition: playground-series-s3e12
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s3e12/discussion/402403

# Hello Kaggle,

This is my first competition and happily finished fifth, even if I highly suspect there's some beginner's luck here. Also, as discussed elsewhere, the private LB may not reflect well CV scores. A fellow kaggler published <a href="https://www.kaggle.com/competitions/playground-series-s3e12/discussion/402357">here</a> despite a high CV score of **0.841**, he ranked **182** on the private LB, and as we'll see later my CV score wasn't that high.

# Helpful posts to check out
The following posts were really helpful and I learnt a lot of stuff through them, especially how to CV properly, when to use train_test_split vs KFold and the importance of trusting CV scores. I highly recommend them to all beginners who want to build a solid foundation for playground challenges.
https://www.kaggle.com/competitions/playground-series-s3e12/discussion/401113
https://www.kaggle.com/competitions/playground-series-s3e12/discussion/399412
https://www.kaggle.com/competitions/playground-series-s3e12/discussion/401344
https://www.kaggle.com/competitions/playground-series-s3e12/discussion/400837
https://www.kaggle.com/competitions/playground-series-s3e12/discussion/400152

# My strategy
**My complete notebook, as I submitted it, is available <a href="https://www.kaggle.com/code/antoinerogeau/ensemble-xgb-lr-rf-knn">here</a>.**
My strategy for this competition was to select a few basic models, finetune them with optuna using RSKfold, have a look at each feature's importance and drop non important important features. Following this, I checked CV scores to make sure they were higher without non important features. I did this for a random forest, xgboost and logistic regression and also added @ambrosm's <a href="https://www.kaggle.com/competitions/playground-series-s3e12/discussion/401344">KNN model</a>. I then built a soft voting ensemble classifier which showed a CV score of around **0.817** if I remember correctly, which highlights once again the limits of the private LB. To be noted, the standalone RF model had a slightly better CV. So I submitted predictions from both.
All in all, as you can see, I don't feel the fifth place is deserved, but this is still rewarding and good for motivation. I'd be highly thankful if you have any comments or suggestions about the way to go forward from here.

Cheers
