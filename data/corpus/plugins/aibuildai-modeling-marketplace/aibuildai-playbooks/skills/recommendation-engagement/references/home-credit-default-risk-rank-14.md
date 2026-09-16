# #14 solution

Competition: home-credit-default-risk
Rank: #14
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64502

Hi everyone,

First of all, never imagine that I can write something as a recipient of gold medal. Thanks to my great teammates [@Adri][1] [@Tomoyo][2].

I just want to share some key features in our solution.

My model has local cv 799+. I used kernel from [@Aguiar][3] as a baseline. 

The features boosted my cv:

TARGET ENCODING: I used target encoding for the categorical features in Application, Bureau and Prev_application tables. In Bureau and Prev_application tables, the first thing you need to do is that creat a map between SK_ID_CURR and TARGET, then map it to Bureau and Prev_application tables. Since one SK_ID_CURR can correspond to many SK_ID_BUREAU or SK_ID_PREV, I used max, min and mean to aggregate the features after target encoding.

P.S. I didn't bother to choose between TARGET ENCODING and ONE-HOT ENCODING. Actually, I used both of them. I think they have information from different perspectives.

TIME WINDOW: I used this feature in every table except Application. For example, in Bureau table, I used 'DAYS_CREDIT' as a threshold for selecting rows. After selecting rows, I aggregated those features again. This approach will generate hundreds of features. I used lightGBM feature importance to select 60% of them(DON'T ASK ME WHY. I DON'T KNOW. XP ). And this feature selection approach can boost my cv further.

One key feature: 

bureau['AMT_CREDIT_DEBT_RATE'] = bureau['AMT_CREDIT_SUM_DEBT']/(1 + bureau['AMT_CREDIT_SUM'])

My teammate, Tomoyo, told me about this feature. It boosted my cv from 798 to 799+

About stacking:

We generated many models(dart, goss, catboost, xgb) with different features. When making meta-model, I used lightGBM and ElasticNet to make 30 oof predictions with different random seeds. Averaging these 30 predictions has our best private score. But we didn't select it. XD 
When using lightGBM, I also selected 90 features by feature importance from first layer model and added them to meta-model as features.


Best,

seaguII


  [1]: https://www.kaggle.com/tita1708
  [2]: https://www.kaggle.com/guziye
  [3]: https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
