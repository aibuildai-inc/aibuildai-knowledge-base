# 2nd place solution

Competition: data-science-bowl-2019
Rank: #2
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127388

First of all, thanks to Booz Allen Hamilton and Kaggle team for such an interesting competition. And congratulations to all the winning teams and all the Kagglers who have worked hard and learned a lot throughout this competition. 

We ranked 38th in Public and 2nd in Private. These final results excited us and one of our teammates, @tiginkgo, has become a new Kaggle master :)

## Results
The best model we chose achieved 0.563 for Public and also 0.563 for Private. 

## Feature Engineering 
**Word2Vec features of title series**
- Considering the series of course titles up to the target assessment as a document, processed them with word2vec and calculated the stats (mean/std/max/min) of the obtained vector.

**Historical feature**
- Count of (session, world, types, title, event\_id, event\_code) as historical data, grouped by (all, treetop, magma, crystal).
- Count, mean, max of (event\_round, game\_time, event\_count).

**Decayed historical feature**
- Historical data decayed for (title, type, world, event\_id, event\_code).
- Decrease accumulation by half for each session.

**Density of historical feature**
- The density of historical data for (title, type, world, event\_id, event\_code).
- Density = (count) / (elapsed days from a first activated day).

**Lagged Assessment**
- Lots of stats (mean/std/...) of num\_correct, num\_incorrect, accuracy, accuracy\_group.
- The difference of hours from the past assessment.
- Per full assessments, and per title assessments.

**Meta Features**
- In order to denote “How having a game_session in advance can lead to an assessment result”, we created “meta target features” for each assessment title. We used oof for train data and KFold averages for the other data such as records without test or meta target.



## Feature Selection 
- Delete duplicate columns.
- Delete high-correlated columns (over 0.99).
- Finally, fetch top 300 features scored by null importance.

## Modeling
- For the validation set, we resampled to ensure one sample per one user.
- StratifiedGroupKFold, 5-fold.
- RSA (5 random seed) of LGB, CB, and NN.

## Post Processing
- Ensemble = 0.5 * LGB + 0.2 * CB + 0.3 * NN.
- Set the threshold to optimize cv qwk.

## Special thanks
The 7th place solution of Elo Merchant Category Recommendation Competition gave us great inspiration, especially for our word2vec and meta features, which were very important parts of our solution.

We are deeply grateful to @senkin13 and his excellent explanations are here:
https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82055
https://www.slideshare.net/JinZhan/kaggle-days-tokyo-jin-zhan-204409794
