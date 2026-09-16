# 3rd place solution

Competition: outbrain-click-prediction
Rank: #3
Source: https://www.kaggle.com/c/outbrain-click-prediction/discussion/27923

Congratulations to code monkey, brain-afk and everyone!
And I’d like to thank my teammates carl and Little Boat. I could learn lots of things from them.

As [Little Boat explained](https://www.kaggle.com/c/outbrain-click-prediction/forums/t/27897/congrats-and-solution-sharing?forumMessageId=157062#post157062), our model is ffm models + xgboost models as first layer, and xgboost as second layer.

I would like to explain our first layer models here.

###**Feature extraction**

Following is the list of unobvious features we used.
We made lots of other features which turned out to be useless.
But analyzing large scale user behavior was very interesting to me.

 - page view counts of each user.
 - page view counts of ad landing page.
 - impression counts for each ad_id, landing document_id, campaign_id and advertiser_id.
 - landing page confidence vector, where confidence vector is defined as a vector which element is composed of confidence level from documents_*.csv. This feature is used only for ffm as numeric data.
 - user confidence vector - this is the average of document confidence vector which is viewed by each user. This feature is used only for ffm as numeric data.
 - inner dot product of document confidence vectors of ad impression page and ad landing page .
 - inner dot product of user confidence vector and ad landing page document confidence vector
 - XGB leaf for ffm feature.
 - immediate document viewed after click event.

###**Modeling and Training**

####**FFM**
In addition to [carl’s light-ffm](https://www.kaggle.com/c/outbrain-click-prediction/forums/t/27892/introducing-light-ffm-and-stack-nn), we used another customized libffm which has pair-wise rank and can take sample weight.
Best public LB score is 0.6974.

####**XGB**
 We used xgboost with rank:pairwise objective.
Best public LB score is 0.6885.

####**Leak Row Excluding**
We trained ffm models without leak row by removing them from training data first, then blend/merge leak information later.
This improved single model score.

####**Present/Future split**
We split test data into ‘present’ data (in time samples) and ‘future’ data (out time samples), then built models for each data.
Here, Features and hyper parameters for each model are optimized independently.
For example we used timestamp feature for ‘present’ model, but did not used it for ‘future’ model. And we used sample weight of training data for ‘future’ model to add higher weight to last day than first day.

Please feel free to ask any questions.
