# 25th Place Solution

Competition: ashrae-energy-prediction
Rank: #25
Source: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/125307

Hey Dear Kagglers :)

First of all, congratulations to all winners and participants!
And many thanks to all contributors for their amazing kernels and topics!

Wanted to put my 2 cents in - so you can find a brief overview of my solution below.

## Motivation
Started work on the competition quite late - just two weeks before the end and my main motivation was to experiment around the use of deep learning with tabular data (and for this kind of problems).

## Architecture
Like probably most of the participants I ended up with 2 layer learning architecture: base layer and ensemble.

### Base layer:
#### Feature engineering
Because of my late start, I didn't really have the time for something fancy (FE is usually my main focus. Not this time though :smile: ).
I ended up with 4 different datasets. On all, I applied cleansing (filtered out the bad rows) as per this great kernel (please upvote: https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks).
Used interpolation to fill in the missing values (on all), added "is_missing" features (to one of the sets).
In addition generated some:
* weather-based features (aggregates/grouping, lag/rolling (moving averages/max/min etc))
* time-based features (including holidays for one of the sets)
* target encodings (in one of the sets).

I draw inspiration from many great kernels in the process - check the list below and please go and upvote the amazing work of the authors - they deserve it!

#### Models
My main focus was here and my goal was to have as diverse models as possible.
I did NOT touche the leak data in any of the models of the base layer.
Trained a total of 19 models. Amongst those:
##### Deep NNs 
As I mentioned my goal was mostly to play around these - so I put significant effort here. My best NNs didn't disappoint - they were approaching the performance of the best public LGBMs (1.09 LB) - but of course, had a very different "point of view" on the problem. That made them extremely useful in the final ensemble, and to that, I mostly attribute my good final score. Some things that worked great here:
* entity embeddings for the categorical variables
* Radam optimizer (that came as a bit of surprise - Radam didn't appear to make a huge difference for me on CV competitions - but here it showed significant advantage. Maybe the size of the network, or the complexity of the problem ... will dig deeper into that for sure)
* Adam with CyclicLR + ReduceLROnPlateau schedulers + longer training
* weather lag features + (few) time features
Things that didn't work:
* more features (not surprised)
* "is_missing" features (kind of surprised)

In short - very happy with my findings. Deep NNs will definitely be part of my considerations in future tabular data competitions.
 
##### LGBMs (of course :smile: ):
* models per site, meter, half-half and on all data - trained on different datasets.
* my best scoring model was per site (performing significantly better than the best public models (1.072 LB))

##### L1, L2 regression models - just to have one more "opinion":
* per site and all data
* made sense for this more linear kind of a problem.
 
### Ensemble
The second major thing that boosted my score.
Based on my previous experience, stacking works better than blending when testing data is not cardinally different (and that can be argued of course).
(NOT) very small disclaimer here:
* this is not valid in all cases
* well... how similar is that test data really...
* how proper your base layer is - is it diverse enough? Are you overfitting there already? Do you introduce leak in the meta model training data if use diverse validation/oof prediction schemes for the base models?
* there are also many other (important) variables - like what algorithm will be used for the meta learner, how it will be tuned, validated etc.
* very risky approach in general (overfitting is BIG concern...).

Still ... it worked well for me in the past.
And worked this time as well.
Let me give you some numbers:
* my best stack: 1.245 (PL)
* my best blend: 1.270 (PL)

**What worked well here:** simple stuff (GLM, Gaussian, 5-fold CV, some regularization, adding meter to the training set). Performed very well for me also in the past (Elo Merchant Category Recommendation competition for example).
**What didn't work that great:** more sophisticated stuff - Random Forest and even XGB, LGBM. All tend to overfit on my base data...

These are pretty much the points, I want to make.
One more time - please spend a minute to upvote the amazing work of the contributors below.
And upvote this, in case you find it useful :smile:
Cheers folks and "see" you on the next competition.

Great kernels that deserve your vote:
https://www.kaggle.com/rohanrao/ashrae-divide-and-conquer
https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type
https://www.kaggle.com/isaienkov/keras-nn-with-embeddings-for-cat-features-1-15
https://www.kaggle.com/roydatascience/ashrae-energy-prediction-using-stratified-kfold
https://www.kaggle.com/iwatatakuya/ashrae-kfold-lightgbm-without-building-id
https://www.kaggle.com/kailex/ac-dc
https://www.kaggle.com/rohanrao/ashrae-half-and-half
https://www.kaggle.com/nz0722/aligned-timestamp-lgbm-by-meter-type
https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08
https://www.kaggle.com/mimoudata/ashrae-lightgbm-without-leak
https://www.kaggle.com/starl1ght/ashrae-stacked-regression-lasso-ridge-lgbm
https://www.kaggle.com/tunguz/ashrae-histgradientboosting
https://www.kaggle.com/yamsam/ashrae-leak-data-station
