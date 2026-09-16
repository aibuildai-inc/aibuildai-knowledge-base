# #1st Place Solution

Competition: allstate-claims-severity
Rank: #1
Source: https://www.kaggle.com/c/allstate-claims-severity/discussion/26416

Hi Guys,

Here is a short summary of my final solution: Nothing extra-oridinary but a lot of diverse models and data preprocessing. So after my 1st submission which scored 1118 on public LB and I landed the 3rd position at the very beginning of the competiton, I thought I would give a try for my solo gold medal here. I had just relocated to a new city and joined a new company at that moment, but I tried my best to keep in touch with the leaderboard progression and hence update my models going ahead.

**Best Single Model:**

My best single model is an XGBoost with the two-way interactions reported on the forum along with some chosen three, four, five and seven-way interactions. It score 1105.12 in the public LB with 10-fold CV around 1124. I chose the higher order interactions primarily based on mutual information. I used [fair_obj][1] with a constant of 0.7 for training that model and `(1 + loss)**0.25` transformation reported by [MrOijer][2].

**My NN architecture**

My best NN scores 1110 in the public leaderboard with a 10-fold cv of 1134.xx. It is very similar to the [public scirpt scoring 1111.8x][3] , just some more neurons in each layer.  I used EarlyStopping, ModelCheckpoint and derived metric to monitor (I used `log(loss + SHIFT)` transformation) and 10-fold 10 times bagged model. Almost all my NN models follow the same architecture.

I used mostly the above two models to train a stacked model (details later in the post) in the second layer, but there are other models with significant contribution. Among them:

**Regularized Greedy Forest**
I had three RGF mixed in to my ensemble trained with slighly different data and loss transformation. My best RGF scores 1113.xx with a cv around 1136.xx. It is a fantastic library that I explored back in BNP Paribas competiton and was determined to use it in one of my competitions. Just one setback is that it's a bit on the slower side being single-threaded.

**LightGBM**
Another very fast and efficient gradient boosting library beside XGboost. I loved it. Best single model 1111.xx on public LB.

**Vowpal Wabbit**
Always found it worth giving a shot and yes it beat random forest and extra trees which I never could get executed with `MAE` criterion in sklearn. Best single model 1133.xx with a cv of 1154.xx. I used quadtraic interactions in the categorical namespace.

**LibFM**
Not much from single model, but definitely helped my ensemble. Best model 1166.xx with 10-fold cv of 1188.xx.

**LibFFM**
Almost at par with my VW. Best model 10-fold cv 1158.xx with public LB 1139.xx.

I also used RF(R-h20), ExtraTrees(sklearn), glmnet (R) and some other models to diversify my final ensemble but none of them yielded significant individual performance.

**Stacking pipeline**

I used 10-fold stacking in this competiton and tried a number of preprocessing on the data:

 - Different order interactions between categorical features.
 - TF-IDF on categorical features.
 - [Category Embedding][4] with NN and different embedding layer sizes.
 - Standard Scalar/ Minmax scalar
 - Different loss function in xgb incuding count:poisson on the rounded and log   transformed targets. Surprisingly it gave me a public LB of 1168.xx.
 - Trained XGB on the bottom 5 and 10 percentile and top 70, 80 and 90 percentile of data. The motivation was to get a better estimate of the exorbitant loss values.

I had a total of 81 models in the 1st level and trained an XGB and two NN on them on the second layer.

**Best Second Level Model**
My best L2 model has a 10-fold cv of 1114.645 and public LB 1097.87.

**Final Submission**

My final submission is an wegthed average of the following: 
    

 1. `w1*NN1^w2 + w3*NN2^w4 + w5*XGB1^w6 + w7` - weights optimized by using `optim` (Nelder-Mead) in a 1-fold manner => apply weights to test predictions => average 10 test predictions for 10x optimized weights.
 2. If NN1 < w1 , then w2*NN1^w3 + w4 Else if  NN1 > w5, then w6*NN1^w7
            + w8 Else NN1

I extend my congratulations to all other winners and my sincere thanks to Kaggle and Allstate group for arranging this exciting competition. 

Special thanks to Scirpus, Vladimir, Tilli, Lauea (as always), Danijel, d3miekno and all the other people who kept the forum so active all the time with their brilliant ideas and scripts :)

  [1]: https://www.kaggle.com/c/allstate-claims-severity/forums/t/24520/effect-of-mae?forumMessageId=140255#post140255
  [2]: https://www.kaggle.com/mrooijer
  [3]: https://www.kaggle.com/mtinti/allstate-claims-severity/keras-starter-with-bagging-1111-84364/code
  [4]: https://github.com/entron/entity-embedding-rossmann
