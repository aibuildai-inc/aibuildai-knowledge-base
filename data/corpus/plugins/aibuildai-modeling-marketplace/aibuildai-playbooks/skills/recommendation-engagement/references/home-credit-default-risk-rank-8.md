# 8th Solution Overview

Competition: home-credit-default-risk
Rank: #8
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64474

First of all, I would like to thank Home Credit and Kaggle for hosting such an interesting competition! The data is very rich and the quality is good, which give lots of ways to do feature engineering. 

I would also like to thank my awesome teammates, [30CrMnSiA][1], [Jin Wang][2], [piupiu][3], [plantsgo][4], [senkin13][5] and [Yuanhao][6] (ordered alphabetically)! You guys are truly talented and amazing kagglers! Our team name “七上八下” means butterflies in my stomach, this reflects our feeling in the last days. 

**Summary**

I think our approach is not different than what other top teams did. Our single models are all lgb models. Personally, I felt the most difficult thing in this competition is the gap between local CV and LB scores. Only 20% data is used for public leaderboard, which makes the LB scores unreliable. On the other hand, the difference between train and test data makes people to hold doubt regarding the CV scores. Hence, we decide to build two finale models: one for best local CV and one for best LB. 

I think the key for our team is that everyone brings a good model that is different from others. We have different approaches to do the feature engineering, which provides lots of diverse models for stacking. I will share my experience in implementing my model (lgb, best public score: 0.809) in the following. The final results show that golden rule still holds: **trust your local CV**. We are using stratified 10-fold as the cross validation method. The local CV is surprisingly stable, the gap (CV - PB) is from **-0.0005** to **0.0003**.

**My solo approach**

I built my baseline model by taking all the insights from the public kernels. This model has a local CV of 0.794x, LB 0.802.
I was checking my credit report one day and realized the bureau calculates the total amount of credit of a person over all the credit cards. Inspired by this, I grouped all the tables with transaction data (credit card, pos cash, instal payments and bureau balance) by both time and ‘SK_ID_CURR’. This gives an overall description about the applicants for every month/installment. 
One more point inspired by the credit report is that I realize the industry buckets DPD into 30/60/90/120, this is also implied in the description of bureau balance. I applied similar approach to all the DPD data.  
I realized the overlap between credit card and application is small, a simple merge would leave lots of NaN values. I decided to fill this NaN by ‘0’. This strategy is applied to fill all the NaN values generated during the merge process.
The above 3 steps are my 2nd round feature engineering. After these, my model reaches a local CV of 0.797x, LB 0.804. 

**After teaming up**

After we formed the team, we realize that the best way to improve scores is adding groups of engineered features together. Since our feature engineering approaches are different from each other, we start to build ‘big’ single models using **combined features brought by everyone**. Interestingly, we found  **treating the OOFs from single tables as meta features**  and putting back to the lgb model could improve both CV and LB significantly. By putting my teammates’ features and single table OOFS, I got two good models: one with CV 0.800x, LB 0.808; the other one with CV 0.800x, LB 0.809.

**Stacking/blending**

The **trick** about stacking/blending in this competition is that all single model predictions need to do a rank percentage before putting them into the ensemble. We think this is because the evaluation metric for this competition is area under the ROC curve, which is sensitive to the relative rank. Before realizing this, my staking/blending CV and LB are always worse than the best single model put into the ensemble. 

With this gold medal, congrats to my teammate [piupiu][7] to become a grandmaster! 


  [1]: https://www.kaggle.com/h4211819
  [2]: https://www.kaggle.com/wangking
  [3]: https://www.kaggle.com/pureheart
  [4]: https://www.kaggle.com/plantsgo
  [5]: https://www.kaggle.com/senkin13
  [6]: https://www.kaggle.com/wuyhbb
  [7]: https://www.kaggle.com/pureheart
