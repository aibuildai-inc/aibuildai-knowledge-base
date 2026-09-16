# 12th Place Approach

Competition: two-sigma-connect-rental-listing-inquiries
Rank: #12
Source: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32118

I have already shared the description of my best single model in another thread. It scores 0.50460 in the private LB. It is XGBoost with trees of depth 4 and about 140 features. The features are the following:
## Features ##
 - Initial variables like number of bedrooms/bathrooms, price, number of images/features, listing\_id itself, etc. Also, label-encoded categoricals like building\_id, manager\_id, manager+building etc.
 - Some metadata from the images like avg RGB components, avg width/height, etc. And, of course, the leak (thanks to @KazAnova) with 3 additional binary variables indicating time blocks of image creations (first 4 days, last day and the rest).
 - String distance between street\_address and display\_address as mentioned [here][1].
 - Regularized price per bedroom-bathroom as mentioned [here][2].
 - From the description: number of words, flag 'is it empty' and flag 'is it website\_redacted'
 - From the created variable: One-hot encoded weekday and month of the listing created time. Sin and Cos of the hour created (in order to reflect that 0 follows 23)
 - From the features variable: just binary variables for top-40 features without any preprocessing
 - From the lat-lon data: 2 sets of label-encoded clusters from DBSCAN with 15 and 50 clusters. Distance from the listing to the top-5 cluster centers in terms of number of observations. So, actually it's a distance to the most dense areas.
 - Various statistics grouped by manager/building/display\_address/DBSCAN clusters, like: min/max/median price; number of managers/buildings per geo-cluster; number of geo-clusters per manager/building, etc. Overall, about 25 variables.
 - And finally the block of Bayessian features, which gave the most significant boost. As was mentioned in the [first posts][3], we can calculate some kind of the rating giving 0 for the low interest, 1 for the medium and 2 for the high. I've constructed such ratings with the regularization (based on the simplified version of [it is lit by @Branden][4]). It's described in the [paper attached by @Branden][5] in the 7th formula. The ratings have been constructed for manager; building; DBSCAN clusters; hour created; first feature in the features list; and some combos like manager+month; manager+day of year. Overall, 8 variables.
 - In addition, inspired by the @gdy5 [script][6], for each manager I've also added three features indicating median price per bedroom in terms of each interest level.

I've also tried lots of other stuff without any success. Unfortunately, I haven't managed to extract any useful features from the groups of duplicated listings (@KazAnova finds them very powerful).
## Stacking ##
Having pretty much good single model, my 1st level models were not enough diverse. There were about 15 models: 

 - Some treating interest level as continuous variable (0 for low; 1 for medium and 2 for high)
 - Some binary classifications for one-vs-all for each interest level

These 15 models were put to 2 second level models, namely XGBoost and LightGBM.
The final submission was just a geometric mean of these 2 models with a [better prior][7] correction. 


Once again, thanks to TwoSigma and all of you guys for the great experience and the nice competition!

 


  [1]: https://www.kaggle.com/ivanoliveri/two-sigma-connect-rental-listing-inquiries/new-features-from-addresses-fields/notebook
  [2]: https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/a-proxy-for-sqft-and-the-interest-on-1-2-baths
  [3]: https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/improve-perfomances-using-manager-features
  [4]: https://www.kaggle.com/brandenkmurray/two-sigma-connect-rental-listing-inquiries/it-is-lit
  [5]: http://helios.mm.di.uoa.gr/~rouvas/ssi/sigkdd/sigkdd.vol3.1/barreca.ps
  [6]: https://www.kaggle.com/guoday/two-sigma-connect-rental-listing-inquiries/cv-statistics-better-parameters-and-explaination
  [7]: https://www.kaggle.com/cpmpml/two-sigma-connect-rental-listing-inquiries/better-prior
