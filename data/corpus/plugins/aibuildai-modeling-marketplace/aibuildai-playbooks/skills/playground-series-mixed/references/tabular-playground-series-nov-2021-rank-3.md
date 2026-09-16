# #3 Solution: Don't trust the cv scores

Competition: tabular-playground-series-nov-2021
Rank: #3
Source: https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291766

Every competition is special somehow, and so was this one. Two observations were most important for my strategy:

1. Training and test data of this competition live in different regions of the feature space. If a classifier can be trained to [classify samples into training and test](https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291003), cv scores may be unreliable. I experimented with weighted training data so that training samples which are near the test region got higher weight than training samples which are far from the test region. The result on the test set (i.e. leaderboard score) was disappointing.

2. After it became known that the [data is chunked](https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/286731), I changed my [cross-validation strategy to GroupKFold](https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/290810). This means that I trained estimators on nine chunks and validated the result on the tenth one. I experimented with a broad spectrum of estimators: linear regression, kernel ridge regression, 1000 nearest neighbors, LightGBM and neural networks. They all lost against a pair of DummyRegressors trained on the leaked data. If nine training chunks don't give any information about the tenth one, why would the ten training chunks give any information about the test chunks?

Based on these two observations, I decided to no longer look at cv scores, but only trust the public lb. In every competition we warn one another not to trust lb scores, but this competition is different: GroupKFold cv scores are useless, stratified cv scores are even more useless, the public lb was the best we could get.

I [probed the leaderboard](https://www.kaggle.com/ambrosm/tpsnov21-012-leaderboard-probing) for the target probabilities of 18 half-chunks: The test data is first split into nine chunks of size 60000 each, and then every chunk is split into two half-chunks at the hyperplane given by the leaked training data. Every half-chunk took one submission - this means that at five submissions per day, 18 half-chunks took four days. From the 18 public lb auc scores I calculated the target probabilities for every half-chunk. @safavieh explains the math in detail in [his notebook](https://www.kaggle.com/safavieh/probing-test-set-chunks-with-math). 

Finally I used up my remaining submissions to determine the best weights for blending the 18 half-chunk probabilities with two public notebooks. The half-chunk probabilities get weighted at 92 % and the public notebooks at 8 %. My final submission is [here](https://www.kaggle.com/ambrosm/tpsnov21-012c-leaderboard-probing).

I thank all people who shared their insights in the competition forum and particularly those who pointed out where my statements were vague and so helped me to get a clearer picture of the facts.
