# 19th Place Solution summary

Competition: ieee-fraud-detection
Rank: #19
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111489

## **Introduction**

First of all, I would like to thank Kaggle and host for hosting such an interesting competition! 
I would also like to thank my awesome teammates, [e-toppo](https://www.kaggle.com/masatomatsui), [kuripical](https://www.kaggle.com/kurupical), [pondel](https://www.kaggle.com/shogonagano).

Our team name ‘クソザコやねん’ means  ‘We are very very very weak’ in Japanese. But, in the end of the competition, our team turned out to be a very strong team, and finally got a gold medal. That’s why we changed the team name to ‘クソザコちゃうねん’ which means  ‘We are not weak’.

By the way, we met each other in the on-site competition called [atmaCup](https://atma-cup.atma.co.jp/), and formed a team at a social gathering for this competition. The next atmaCup will be held in Osaka, Japan on November 23, 2019. Let's participate with us!

## **Model pipeline**



## **Feature Engineering**

- UserID (Combining card features with the date D1 days before the TransactionDT)
- Aggregaion by UserID (count,mean,var)
　- C1-C14
　- D1-D15
　- TransactionAmt  etc.
- max/min values, max/min columns between similar C,V features
　- example:
          &nbsp; &nbsp; &nbsp; &nbsp;V212, V274, V317
          &nbsp; &nbsp; &nbsp; &nbsp;V213, V275, V318
          &nbsp; &nbsp; &nbsp; &nbsp;V167~V171, C2
- decompose D, C, V feature (use PCA)
- nan pattern(D, C, V, M)
- isDecember Flag

## **Validation Strategy** 

We first used stratified 5 fold with no shuffle for validation ([kurupical](https://www.kaggle.com/kurupical) kfold instead of Stratified). 
Then we started using time series splits to avoid huge shakes.Our time series splits has 7folds. 
The | indicates the train/val split.
&nbsp;　1) 0 1 2 3 | 4
&nbsp;　2)  0 1 2 3 4 | 5
&nbsp;　3)  0 1 2 3 4 5 | 6
&nbsp;　4)  train all data using 1.1x number of rounds early stopped in 3
Blend: using weighted average 1) * 0.1 + 2) * 0.1 + 3) * 0.25 + 4) * 0.55.

## **Models and ensemble**

In this competition, we decided to choose our final submission with the following 2 different strategy in order to prevent the risk of shake(which didn’t occur).
- submission1: High LB model (Full model in the slide.)
- submission2: Conservative model (sumission of k2_samp model in the slide)

Finally, we made 10 models (we count as the same model for changing the validation method or k2_samp), included Only ProductCD=W models, without MAGIC features, and so on. Our best single model scored LB: 0.9588 by LGBM.

For the optimization of blending weight,  we used Spicy Nelder-Mead which were used in the [2nd place solution of Home Credit Default Risk](https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions?slide=55) . We decide not to optimize the weight of timeserie based model and Stratified based model and chose the following weihgt as the final submission (0.4*timeserise + 0.4*stratified + 0.2*timeseries(k2_samp)).

## **Post Process**
We made unique user ID which is concatination of  card1-card6 and difference between Day-D1, Day-D10, and confirm the count of record and isFraud’s mean.
If the user is all “Fraud” transaction in the train datasets, we replaced the submit prediction to 1, if the user is all “not Fraud” transaction in the train datasets, we replaced the submit prediction to 0. This postprocess boosted +0.0007 in Public, +0.0004 in Private. if you want to know the detail, please refer to my kernel: [https://www.kaggle.com/kurupical/postprocess-19th-place-solution-team](https://www.kaggle.com/kurupical/postprocess-19th-place-solution-team)

## **Conclusion**

In this post, we briefly summarised our model pipeline. Please feel free to ask us for any questions you have for our solutions. 

## **Appendix  (Methods that didin’t work for us)**
- Neural network
- Pseudo labeling
