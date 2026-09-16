# 1st place solution summary

Competition: icdm-2015-drawbridge-cross-device-connections
Rank: #1
Source: https://www.kaggle.com/c/icdm-2015-drawbridge-cross-device-connections/discussion/16122

First off, I'd like to thank the competition hosts at drawbridge for all their work putting this contest together.  Next, I'll thank [DataLab USA][1] for providing the computing resources for our solution.  Finally, thanks to all the competitors who kept things interesting.  I'm looking forward to seeing your approaches.

 - We generated a training set from the data as a collection triples of (device_id, cookie_id, ip) for each instance where a device and cookie appeared on the same ip.
 - To this collection of triples, we join all the basic information about the device, cookie, and ip address.
 - In addition, we generate a few hundred features based on the interaction between device, cookie, and ip.  The most influential of these were ranks of each cookie partitioned by device and ordered by a basic attribute, say idxip_anonymous_c3 or ip_anonymous_c2 for instance.
 - Cookie properties and any categorical features were one-hot encoded and also joined to the training set.  We winnowed the list down substantially through cross-validation and were left with a few hundred in our final model.
 - To this we added out-of-sample predictions of a few xgboost models built on cookie_all_basic to predict whether a cookie was matched to any device in the train set.
 - To reduce the size of the training set we downsampled the cellular ip addresses quite heavily and also dropped any -1 drawbridge handle cookies.
 - On this reduced dataset we built a learning-to-rank model which was a modified version of xgboost's "rank:pairwise" partitioning by device.
 - For each device, we took the drawbridge handle of the highest scoring cookie and submitted all the cookie_ids for that drawbridge handle.

We did attempt some ensembling for a final solution.  It seems this was ill-fated as the private leaderboard score for strongest model was higher than the ensemble.

  

 


  [1]: http://www.datalabusa.com
