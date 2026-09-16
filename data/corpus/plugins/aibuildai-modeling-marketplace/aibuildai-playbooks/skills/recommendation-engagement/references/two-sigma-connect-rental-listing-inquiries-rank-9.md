# 9th Place Solution

Competition: two-sigma-connect-rental-listing-inquiries
Rank: #9
Source: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32146

My solution was a plain three level stack, 17 base models into XGB and Keras stackers and a geometric mean of the two.

I don't have time to write up my features yet, but I will follow up this post tomorrow with [more details][1]. As a sneak preview I've attached a feature importance list from one of my LightGBM classification models, with all the feature names and some measures of importance:

 - Gain - cumulative gain over whole model
 - Count - number of times used
 - Avg - Gain/Count
 - Trees - number of trees that use the feature

Hopefully the feature names are mostly self explanatory.


## Feature Naming Convention

- xxx_count is a count of the # of times the value appears in train & test
- xxx_min xxx_max xxx_mean are list summary features
- xxx_diff means subtraction
- xxx_ratio means division
- f_xxx are one hot coded apartment-feature flags
- xxx_cat_int is a integer coded categorical
- fts is my name for the leak feature ("folder time stamp")

The single best feature is **price_per_sqft** - from [Darnal's excellent breakdown of price & rooms interactions][2].

Following that is **cv_price2** - validation predictions of a 5 fold CV LightGBM regressor model of price (trained on log(price) and exp() converted back) based on lat/lon, address, bedrooms, bathrooms, built using all price data (train and test joined). Then **cv_price2_ratio** - listing price/cv_price2 - does this listing look over or under valued?

[More tomorrow!][3]

________________________________________________________

## Level 1 Models

Scores are from 5 fold stratified CV.

**Classifiers (Log loss)**

 * et1	0.58221 (Extra Random Trees)
 * lgb3	0.50573
 * lgb4	0.50627
 * lgb5	0.50582
 * rf1	0.55560 (Random Forest)
 * rte1	0.58279 (*)
 * xgb0	0.53982
 * xgb1	0.50611
 * xgb2	0.50587
 * keras	0.53630 (average of five different NN's)

**Regressors (RMSE)**

 * etr1	0.47359
 * ftrl1	0.51537
 * ftrl2	0.49579
 * lgbr1	0.45111
 * lgbr2	0.44766
 * rfr1	0.47306
 * xgbr1	0.44717

(*) rte is [Gert's "Regularized Tree Ensemble"][4], although I only ran it as a quick test with default settings.

I spent the last two days adding and updating the regressor models, which accounts for the late surge.

The FTRL models are my own adaptation of the original code in Java, added on the last day. They take only a few seconds to train for one fold, so it was very easy to iterate and tune them. Even though I think they fell a long way short on accuracy (due to time), they improved my level 2 CV scores by about 0.003.

ftrl1 was a simple first attempt on some basic features.

To create ftrl2, I wrote a quick forward selection loop: a greedy process of adding every remaining feature one at a time & logging validation error, then adding the best ones in. After that, a loop to randomly create interaction terms over pairs of input features which got it to 0.49579, and bumped up its importance in the L2 XGB model.

________________________________________________________

## Level 2 Models

 - xgb  0.494664 +/- 0.004521  public 0.49723 private 0.49552
 - keras  0.494422 +/- 0.005545  public 0.49782 private 0.49553

**Level 3**

Geometric mean:  public 0.49670 private 0.49470

_________________________________________________________


I must add thanks to, in no particular order:

 - All Kagglers who've posted interesting and thought provoking comments and kernels in the forums. The Kaggle community has a really inspiring culture of sharing.
 - Silogram for his generous [Notes from sub-0.5 solution][5] post.
 - KazAnova for sharing the leak information.
 - All who competed, the leaderboard dynamics are always an interesting aspect of each competition.
 - Two Sigma and RentHop for an interesting, manageable and intuitively understandable data set.


  [1]: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32146#178428
  [2]: https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/a-proxy-for-sqft-and-the-interest-on-1-2-baths
  [3]: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32146#178428
  [4]: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion/20207#115555
  [5]: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/31765
