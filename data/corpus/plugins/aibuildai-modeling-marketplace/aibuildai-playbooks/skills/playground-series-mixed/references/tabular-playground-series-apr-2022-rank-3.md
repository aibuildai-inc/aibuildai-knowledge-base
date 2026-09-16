# #3 Solution

Competition: tabular-playground-series-apr-2022
Rank: #3
Source: https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322269

Hi everyone! As usual, here's a short write-up of our competition solution.

First of all, congratulations to @davidedwards1 and @azzamradman for their #1 and #2 positions respectively. After reading the write-ups, I am impressed by the originality of @davidedwards1's solution and the score of a single model obtained by @azzamradman. Since you both used a blend, I do think we can share another trick that is easy to apply that can further improve your score! Feel free to try it out and let us know if it works :).

Of course, a huge thank you to my two bright team mates (and colleagues) @moeflon and @jeroenvdd for their effort and input. It was a lot of fun to compete with the two of you. This was also their first Kaggle competition (that was not in-class)! Our team name (VD Brothers) comes from a similarity between our names: Jeroen **V**an **D**er Donckt, Gilles **V**an**d**ewiele and **V**ic **D**egraeve.

I think I'll keep this write-up rather brief, as the largest part of our solution has actually been [publicly posted by my team mate @jeroenvdd](https://www.kaggle.com/code/jeroenvdd/tpsapr22-best-non-dl-model-tsflex-powershap). This was also the main motivation for us to participate in this competition: try out some of our in-house software packages and share it with the Kaggle community ( [tsflex](https://github.com/predict-idlab/tsflex) and [powershap](https://github.com/predict-idlab/PowerSHAP) ). A shout-out must be given to @ambrosm as [his notebook](https://www.kaggle.com/code/ambrosm/tpsapr22-best-model-without-nn) served as a baseline for us during this competition.

# The usual suspects (subjects)

We used 10 GroupKFold for all of our trained models, based on the `subject` identifier. This resulted in CV scores that corresponded very well to LB scores.

# The shape of you

Our team member @moeflon extended the feature-based approach by a unique type of features that have not been used much before here on Kaggle: [shapelets](https://tslearn.readthedocs.io/en/stable/user_guide/shapelets.html). These are small subsequences that are predictive for a certain (group of) class. They can be mined through gradient descent, i.e. deep learning. The best public implementation available can be found in [tslearn](https://github.com/tslearn-team/tslearn/) but Vic ported the code from keras to torch to make some extensions: learning rate scheduling, early stopping and the ability to pass along extra features. Especially this final extension is valuable since the shapelet mining was performed **AFTER** our feature extraction. By passing along the features we already extracted, we force the network to learn shapelets that are complementary to the features we already extract. Our best submission, with these shapelets, scores `0.98052` public LB, `0.97706` private LB.

# Our milkshake brings all models to the yard

The extra percentage of AUC (`0.98052` -> `0.99037` public score) was obtained by stacking together different models (and fitting a catboost model on it):

- Some older versions of our feature-based approaches
- https://www.kaggle.com/code/dlaststark/tps-apr22-tfv2
- https://www.kaggle.com/code/dlaststark/tps-apr22-tfv1
- https://www.kaggle.com/code/bannourchaker/deep-learing-part3-cnn-inceptiontime-con11
- https://www.kaggle.com/code/bannourchaker/deep-learing-part4-hybrid-cnn-lstm-parallel-con166
- https://www.kaggle.com/code/bannourchaker/deep-learing-part2-bilstm-densenet-rnn-con6
- https://www.kaggle.com/code/davidedwards1/tps-april-tensorflow-bi-lstm-10f-spatialdropout
- https://www.kaggle.com/code/siukeitin/tps042022-baseline-xgboost-model-on-2500-features

**For most notebooks we only had to make a few changes such as storing the oofs during cross-validation, doing group KFold but actually on subject (many did it on sequence), reset the model in every fold instead of iteratively fitting it over folds, etc.**

**A nice & easy trick that boosted our CV/LB score with 0.002 was to group our predictions by subject and calculate aggregates (mean, min, max, std, ...) from that!**

# Things that did not work

We tried quite a few things that did not work out in the end:
- Pseudo-labeling the confident predictions in the test set.
- Sequential models on the sequences of the same subject (order based on sequence id). There does seem to be a sequential nature in there (i.e. the previous or next values are more often the same than different). We tried fitting Hidden Markov Models one each subject but to no avail. We were, however, able to marginally improve if we estimated the HMM parameters based on the label data for the training set, but we couldn't find a way to find these accurately without label info.
- Many types of features, which we mostly were able to automatically exclude with the help of [powershap](https://github.com/predict-idlab/PowerSHAP)
- Trying to stitch together sequences of the same subject.
- Fitting a model that predicts whether a subject only has sequences with state 0 (as a post-processing step). About 9% of the subjects (or almost 5% of the data) contain sequences with only 0's. A model that achieves AUC 1.0 can be fit to predict these subjects. However, adding this model does not bring any added value (our models already implicitly learned these things).
- ...



Et voilà! This was hopefully somewhat useful. In case you have any questions, or would like to know more information about a specific part/step of our solution, feel free to ask below! I might edit this post if I think of extra relevant information.

Gilles, Jeroen, Vic
