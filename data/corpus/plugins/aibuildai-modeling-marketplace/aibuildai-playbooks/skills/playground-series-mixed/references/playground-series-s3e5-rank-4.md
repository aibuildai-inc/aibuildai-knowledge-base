# 4th Place Solution

Competition: playground-series-s3e5
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s3e5/discussion/386645

Hi Everyone! Thanks again to all the competitors and to the organizers at Kaggle for another fun round in the Playground Series. As we all somewhat expected, there was another big shakeup when the private LB was revealed. This is somewhat unsurprising, given that:

1) There were 3 `quality` classes that were quite rare in the dataset (I'll refer to them as rare cases below) - 3, 4 and 8; and,
2) We had a metric of quadratic weighted kappa that heavily penalized ratings the more that they drift apart from each other. 

What these two conditions meant is that small differences in classifier performance could have a large impact on the overall public LB scores. As pointed out by @ambrosm in the discussion thread [Is this competition a lottery?](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/383429) moving a single quality rating one step closer to the rating it should be has a huge impact on the quadratic weighted kappa metric. And of course, this means the opposite is true too - moving it one step further away also has a huge negative impact. This means being conservative and staying close to the center of the ratings scale most of the time is better than taking a chance by making more predictions at the extreme ends of the scale, given that most of the data was clustered in the middle of the scale. In other words, while the latter approach may hit the correct rating of 3 or 8 some of the time, the hit that it takes from being in greater disagreement more often for misclassifications of items that were supposed to be near the center simply outweighs the benefit of being right once or twice for the rare cases. More on that in a bit.

For this competition I built a total of 1,466 models that included first level models such as CatBoost, XGBoost, LightGBM, various neural network configurations, and second level stacks that used simple ridge regressors and stochastic gradient descent. [My EDA discovered](https://www.kaggle.com/code/craigmthomas/play-s3e5-eda-models) that regressor models worked better than classifiers, and that using a simple optimized rounding strategy as initially implemented in [the notebook](https://www.kaggle.com/code/paddykb/ps-s3e5-regression-optimise-class-cutoff) by @paddykb for determining class threshold was paramount. My EDA also uncovered the fact that simple feature engineering based on correlation worked best. Overall, my best model was a ridge regression based on a stack of 25 first level models types such as XGBoost, LightGBM, and CatBoost with differently tuned hyper parameters. Each first level model was built using a mixture of the competition and original data, and used a minimal amount of feature engineering (more on that below). This model had the lowest gap between my own local metrics and the public LB, and both the public LB and my local metrics trended upwards (in other words, there was no evidence that I was overfitting the public LB or local metrics as well). 

Some of the learning elements that I was interested in for this competition played a key role in my data exploration and final model selection. For this round of the Playground series, I was interested in several different aspects of the dataset that we were using, and concentrated on four problem areas:

1) The validity of mixing original + competition data.
2) The usefulness of engineered features.
3) The utility of stacking models.
4) The impact of rare case misclassifications.


#### Mixing Original + Competition Data

For the first item - mixing original + competition data, I used adversarial validation as a rough metric as to whether we should mix them (there were several others who did this as well - @kdmitrie [posted a discussion thread](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382545) very early on). [My own adversarial validation exploration](https://www.kaggle.com/code/craigmthomas/play-s3e5-eda-models#1.7---Original-Data) showed that the AUC ROC score of a classifier trained to detect differences between the two sets was 0.6321 (with [duplicates removed](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/383737) - shoutout to @mattop for the duplicate discovery). This suggested that there were detectable differences between the datasets, but not so great as we saw in previous competitions. As others had demonstrated in their notebooks, the best way to account for the differences is to train using the mix of datasets, but only produce cross-validated metrics that use the competition data. In this case, I found that the mixture of datasets had better local CV metrics than the competition data alone.


#### Engineered Features

There were many suggestions on how various features should be combined (a very interesting suggestion came from @phongnguyen1 where they [asked ChatGPT to suggest features](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/383685)). For my approach, I concentrated mainly on engineering that combined features based on Spearman Correlation such that when we quantized the engineered feature, we could essentially "spread out" the quality rankings so that at most 2 or 3 of them appear in each discrete bin, rather than seeing 4 or more appear in the same one. A good example of this was with the `density` feature. When density is quantized, and ratings are counted and placed into each discrete bin, you end up with:



If however, you divide `density` by `alcohol` - the features share a fairly strong negative correlation - you end up spreading out the quality ratings so that fewer of them occur together in each discrete bin:



This spreading out of the quality ratings theoretically gives an edge to gradient boosting tree approaches. Overall, I empirically tested my engineered features in [my EDA](https://www.kaggle.com/code/craigmthomas/play-s3e5-eda-models#3.x---Model-Comparison). The only two sets of engineered features which really provided lift were those that explored variations on acidity and density, as you can see below:



#### Stacking Models

The previous episodes in this year's Playground Series were very surprising to me. Model stacks or ensembles appeared to perform much worse than simple, single models. This is surprising in that most other competitions I've taken part in have proven over and over that model stacks are far superior, and are usually the safer bet when it is time for final submissions. Given that we had a new metric to optimize, I decided to revisit stacks. In nearly all instances, my model stacks provided much better performance when compared to the single model - except for some of my CatBoost models. Some of the CatBoost models had local metrics that slightly outperformed the stacks, but once again, the inversion between my local metrics and the public LB had me dismiss those models as being overfit. Ironically, one of my dismissed CatBoost models with the best local metrics landed a private LB score of 0.60201 - a first place standing. Once again however, I conclude that there is no way that I would have chosen that model based on the metrics I observed - I would have needed to gamble on my second submission option, which I had reserved for another experiment regarding rare case misclassifications (see next section below).


#### Rare Case Misclassifications

As I mentioned above, rare cases were key to the competition, and to how we were to trust (or not trust) our performance based on public LB scores. Moving a rating one spot higher or lower resulted in massive shifts in the leaderboard. Given that such massive shifts were possible, I stopped placing so much importance on public LB standing as to where I was actually going to stand in the final reveal, and instead used it as a general barometer. In other words, my local metrics had to trend in the same direction as the public LB.

As part of my machine learning framework, I made sure to generate a confusion matrix with each model, so I could observe what "good" models were doing, and what "bad" models were doing. In every case, my best models were conservative - they tended to make predictions that were clustered around ratings 5, 6, and 7. The best models however, showed evidence that while they still stuck predictions for class 3 and 4 on to clases 5 and 6, there were more predictions closer to their repsective class than other models. To see what I'm talking about, you can see the confusion matrix below for my top performing model:



As you can see, while it never predicts a single example as being class 4, it puts the bulk of the actual class 4 samples very close to it. The same thing happens with class 8 - while it never actually labels anything with class 8, it puts the bulk of them close to class 8. We can compare that with my second "gamble" submission - a neural network that actually makes predictions of class 4 and 8:



As you can see, it correctly classifies 5 samples as belonging to class 4, but in doing so, misclassifies 34 examples that are actually class 5. Because the quadratic weighted kappa penalizes so heavily when we have disagreements, we take a sizeable hit in our score. I gambled on this model even though my metrics didn't support it as a contender because it was such a large departure in approach from my other submission (i.e. it was a single model, neural net, made predictions of classes 4 and 8, and had at least somewhat decent performance metrics). 


#### Conclusions

Once again, another interesting competition that forced us to really examine the metric at play and how different folds of data (public vs private LB) can have an impact on our final standing. The quadratic weighted kappa, combined with rare cases, made for an interesting competition.
