# Rank 2 approach - brute force ensembling (118 oofs)

Competition: playground-series-s4e12
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s4e12/discussion/554505

After participating in these playground series for the last few months, I am glad to have finally snagged a top 3 finish. These few months have been a great learning experience and I am grateful to all members of the community here that have contributed to our shared knowledge or introduced new insightful techniques.

I will say that this competition felt a bit more straight-foward than usual as the public lb scores and the cv scores lined up almost perfectly (an x decrease in cv score almost always resulted in around x decrease in public lb score) which meant that it required a bit less finesse in choosing between submissions or identifying where your models may have overfit. This was also reflected in the results; there was almost no shakeup at least in the top 10 compared to some of the past competitions.

My overall approach was not novel in any way - it followed the mantra of gathering oof predictions, ensembling, and repeat, which was very similar to [here](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/537029). Additionally, after seeing @cdeotte 's solution, I feel that my solution was very much a 'brute force' approach 😀. Nevertheless, there were a few techniques or tools I used that may of use to others which I will explain below.

**Models and frameworks**
- Automl frameworks are the easiest way to generate many out of fold predictions and I would recommend anyone who hasn't explored the usage of these to try them out in the next competitions. I used quite a few automl frameworks to generate my out of fold predictions, namely autogluon, h2o, FLAML and LAMA. Some notes on these: I found that autogluon generally performed well but was very sensitive to overfitting (a phenomenon called stacked information leakage which is mentioned [here](https://github.com/autogluon/autogluon/issues/2779) by @innixma. This did not occur when using the base features, but when I added a categorical version of healthscore (which was a high cardinality feature) the higher level models experienced this overfitting quite badly. H2o and LAMA performed weaker this time, and the performance didn't reach that of my personally created models even when provided with the same set of features. FLAML pleasantly surprised me with the performance, but this did require longer run times. In the end, to provide FLAML with enough fitting time whilst still only utilising the free kaggle resources, I ended up fitting FLAML five times for each of the five folds concurrently in separate notebooks and combining the predictions afterwards.
- Pytorch Tabnet, which was not great for predicting, but was very useful for ensembling the oofs at the end
- Tensorflow Deeptables, the performance was quite weak for this competition.
- Personal models, nothing special, just implementing various models from different libraries (XGBoost, LGBM, Catboost) to conform to a basic interface which can be found [here](https://github.com/yuwei-1/Kaggle-tools/tree/main) if interested.

**Feature engineering**
- Thanks to @backpaker for contributing his set of features. I ran some feature importance and identified a subset of useful features which I experimented using with various models. Also thanks @swagician for breaking down the various feature engineering techniques of backpacker and identifying that the categorisation of health score was crucial and one the main score improvements for me.
- @backpaker also introduced nonlog catboost oof features (stacking) which provided a slight performance boost, and I tried to further this technique by experimenting by creating another catboost oof which was trained to predict multiclass labels, where the labels correspond to the target feature organised into classes based on the order of the target, i.e., 0 - 9 'Premium Amount' assigned to class 0, 10 - 99 class 1, which also yielded a small performance improvement.
- I used autofe to generate additional features, which resulted in models which achieved similar performance to the the ones trained on backpacker's features. I didn't combine these feature engineering methods, but rather trained the same models on each of these different features to provide some varied predictions which I thought might fare better when it came to ensembling later.

**Ensembling**
- I used a Ridge regressor as my main ensembling technique to decide the weights to assign to each oof prediction, but hillclimbers achieved an identical score with less models (73)
- I also used Tabnet to ensemble the predictions which I then threw in as another oof which did improve the overall ensemble score by providing some non-linear combination of the predictions which proved to be important.

Finally I'd like to note that early on in the discussion forum it appeared as most people had ruled out the use of the original dataset, but upon some experimentation, I found that the original dataset improved the performance of tuned models greatly, which reduced my best models at that point from around a cv score of 1.031 to 1.027.

Please feel free to ask questions where I have been unclear. Thanks for a great year of data science, I look forward to the next - happy new year!

Yuwei (SCRIPTCHEF)
