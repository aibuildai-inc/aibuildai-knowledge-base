# 2nd Place Solution

Competition: two-sigma-connect-rental-listing-inquiries
Rank: #2
Source: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32148

Hey Kagglers,

let me start with some congratulations: Congratz @plantsgo for an impressive performance here. Well done! I further congratulate all the other Top10
finishers, medal collectors and last but not least @KazAnova for being undoubtably one of the MVP's in this competition. StackNet is a striking piece of software, which sets a new baseline in kaggle competitions.  <br>
I also appreciate his move to publish the leak from which I profited without a question. While I like the spirit behind those actions, they almost always leave some people behind unhappy. I feel especially sorry for Silogram, not least because he didn't have the opportunity to "strike back". Besides, I want to note that I really liked the reaction of @Little Boat, who appreciated this "deck shuffling" even though he already had most of the leaky-information in his models. That's an admirable attitude!

----------
Now, let's get technical. :) <br>
For me, it was an intense run that I started a bit more than two weeks ago and I came up with an ensemble of about 60 base models combined with a Keras-NN as 2nd level meta model. 

The public kernels alongside with some insightful forum threads provided me a solid basis to work with and helped me to guide my solution building (for instance not to try to get something out of the images). Hence, my thanks goes to all contributers and in order to name at least a few: @KazAnova, @SRK, @Li Li, @Darnal, @Poonam Ligade, @gdy5, @Branden Murray, @Enrique Pérez Herrero, @den3b, @Mamy Ratsimbazafy, @rakhlin, @the1owl, @CPMP and @Scirpus.

Ensemble Overview
----

**Base Models**
<br>
 - 32 LightGBM models <br>
 - 9 ET models (sklearn) <br> 
 -  7 RF models (sklearn) <br>
 -  5 Keras models  <br>
 -  3 XGBoost models  <br>
 -  @KazAnova's StackNet example base-level predictions

Those models have been trained on different feature sets and they are either multi-class, regression or binary classification models built on 
5-Fold stratified splits.

**Best Base Model**
<br>
My best base model (LightGBM) achieves a CV-loss of 0.50135 and scores 0.50557 on the public and 0.50470 on the private LB. 
I noticed that very small parameter changes led to significant changes (up to 0.004) of the per fold cv-losses and hence I decided to apply
grid search bagging: The first bag was trained with the best parameters found so far. Given that, I performed a grid search in the vicinity of those parameters and checked after each CV pass, if the newly built model improved the overall CV-loss after blending it in with weights choosen according to the current number of bags in the overall blend. That resulted in a 12-bagged model with the scores above. A corresponding "normal" 15 times bagged model with best parameters found scores 0.50632/0.50508 with an CV-loss of 0.50175.

**Meta Modelling**
<br>
My two finally selected submissons are different trained versions of the same 2-layer Keras-NN. The first was trained on a CV-optimized fixed amount of iterations with a full pass for creating test predictions and the second (and better one) was trained using earling stopping with 35 bags per fold. 
The corresponding scores are: 0.49619/0.49450 versus 0.49489/0.49302.


Feature Engineering
-------------------

Feature engineering was by far the most time consuming part for me and I created alot of statistics either based on the entire dataset or based on subgroups like 
for example (manager_id, bathrooms, bedrooms, dayofyear). The most commong statistics are mean prices, differences from the mean prices, price ratios, counts of unique manager_id's, total amount of listings and in
general differences from mean or median feature values for the given group. I also created some timeseries features like target leads, lags and cumsums within groups, creation time deltas, rolling target means etc.

I initially thought, that they would help to better predict the groups of duplicated or very similar listings, but it did not work out. In general, the duplicated listings were much harder to predict and I spent quite some time to find something to close the loss-gap in comparison to the unique listings. Especially the groups of listings which differ only in created time and where all the 3 labels occured were troublesome. 
My best single model scores 0.47400 on unique listings and 0.58954 on duplicates (which accounted for roughly 20% of the dataset).

**Likelihood Features**
<br>
I created quite alot of likelihood estimates with additive smoothing for all high cardinal categoricals, combinations of those and conditioned on time, i.e.
basic ones like the famous p(y | manager_id), something like p(y | manager_id, bathrooms) or p(y | manager_id, dayofyear) etc.
Analogue to the "manager skill" I reduced each to a scalar by applying the dot product with vector [0,1,2]. In order to dodge overfitting issues, I avoided leakage by creating all likelihood features in a 2-nested cross validation. Those features were among the most important and especically the time related ones yielded a nice extra boost over the very well working manager_id lhoods.

**Clustering & Proxies for Points of Interest**
<br>
That was another group of more helpful features. It consists of kmeans cluster of (latitude, longitude) followed by computing statistics like the ones above and cluster center distances.
In order to get some proxies for PoI's in the neighborhood, I created clusters after filtering the dataset based on certain words in the descriptions. That way, I estimated coordinates for things like
"supermarket", "shopping", "subway", "bus", "health", "fitness", "park" etc. Afterwards I created minimal distances to those locations as well as counts based on different distances cut-offs. 

**Simple Text Features**
<br>
.. like uppercase to lowercase ratios, count of characters like '*', '!', '$', '<' etc and flags for the presence of a phone number or email.

**Completeness Score**
<br>
Here I assigned 1 point for each field which was properly filled and the sum of those was the "completeness score" of the corresponding listing.

**XGB Embeddings**
<br>
I trained some very deep xgb models on the 15k most common words in "description" only, on all words of "features" only and on the combination of both. The corresponding predictions have been used as features in some of the base models. 

**Manager Groups**
<br>
There are some descriptions, which were used by different managers and I assigned new common ID's to them, based on the assumption that they belong to the same agencies. This way, the cardinality of manager_id could be reduced by several hundreds, but I did not manage to squeeze much out of it.

*to be continued ...*


**Some Links**
<br>
- My code for stacking / generating oof-predictions: [kaggletils - CrossValidator][1]<br>
- My code for estimating likelihoods : [kaggletils  - LikelihoodEstimator][2]


cheers,<br>
Faron


  [1]: https://github.com/Far0n/kaggletils/blob/master/kaggletils/ensembling/stacking.py
  [2]: https://github.com/Far0n/kaggletils/blob/master/kaggletils/estimators.py
