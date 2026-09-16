# 13th place - time series features

Competition: home-credit-default-risk
Rank: #13
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64593

First of all congrats to the winners and the organises! This was a fun competition and a very popular one with lots of activity on forums and kernels. 

Before I start, you will notice that when I type a link, I post the whole URL - that is because for some reason the `hyperlink` option is not working - not sure why. So I apologise for not being neat :/

In many ways this competition reminded me of porto (https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) where the target variable was quite sparse and cv was not in line with LB for quite some time. At some point our cv was **0.7959** and public  LB was **0.806**. Our latest (stacking) model had cv of **0.803** and public LB again **0.806** (slightly worse than the one above!) .I had a hunch that Michael's technique would work well here too :)  . Maybe your 1st-pace thread here will be even more popular than that one (https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629) which keeps receiving new posts everyday almost!

In our solution we generated over 10,000 features. About 1,000 are of similar nature to the ones included in nenputne-ml solution (https://github.com/neptune-ml/open-solution-home-credit) - thank you very much for sharing. 

The remaining are mostly based on time series features. Its been a while since I last used that much SQL!  We constructed various 96-lag views for various attributes. for example:

 - Sum bureau status by month
 - Sum credit by month
 - Sum Credit to Balance by month (from credit cards) 
 - Sum Instalments per month

   We created around 60 such time series. This is an example of SQL:

```


    create table home.cash_active_cumulative as
    select a.SK_ID_CURR, 
    
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-96 then 1 else 0 end ) as Active_cumulative_count_96,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-95 then 1 else 0 end ) as Active_cumulative_count_95,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-94 then 1 else 0 end ) as Active_cumulative_count_94,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-93 then 1 else 0 end ) as Active_cumulative_count_93,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-92 then 1 else 0 end ) as Active_cumulative_count_92,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-91 then 1 else 0 end ) as Active_cumulative_count_91,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-90 then 1 else 0 end ) as Active_cumulative_count_90,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-89 then 1 else 0 end ) as Active_cumulative_count_89,
    ........
    ........
    ........
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-6 then 1 else 0 end ) as Active_cumulative_count_6,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-5 then 1 else 0 end ) as Active_cumulative_count_5,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-4 then 1 else 0 end ) as Active_cumulative_count_4,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-3 then 1 else 0 end ) as Active_cumulative_count_3,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-2 then 1 else 0 end ) as Active_cumulative_count_2,
    sum(case when a.NAME_CONTRACT_STATUS='Active' and MONTHS_BALANCE=-1 then 1 else 0 end ) as Active_cumulative_count_1
    
    from home.cash as a
    group by a.SK_ID_CURR
    order by a.SK_ID_CURR
 
```    

I have uploaded a small sample file so that you can get the idea. In that file ( ` cash_active_fcumulative_sample.csv` ) , for each `SK_ID_CURR`  and for each past `MONTHS_BALANCE` , for when `NAME_CONTRACT_STATUS='Active' ` , it sums the `CNT_INSTALMENT_FUTURE` . So at any point in time, out of all accounts, how many future instalments are expected to be made?

Based on these 96-lag time-series we generated thousands of  time series features. These included:

 - Moving averages on different lags
 - Other aggregated measures like stds, medians,kurtosis skewness, max, min
 - Exponential smoothing
 - Correlations and regressions with time

and many more. This feature engineering is heavily based on the work we do with Driverless ai (https://www.h2o.ai/products/h2o-driverless-ai/ ) . it is more analytically explained in the following video-talk if someone is interested. (https://www.brighttalk.com/webcast/16463/330616/time-series-in-driverless-ai) 

Modelling-wise, apart from the usual suspects ( Lightgbm, xgboost, vanilla nns) what worked well was stacking (vertically) for each user all these time series and running CNNs or LSTMs . Even with these tricks though, our nns did not exceed 0.794-5 on (public) LB.  

We stacked around 100 models built on 2 validation schemas , one where we average predictions for the test data for each fold (e.g. without retraining on whole training data) and one where we did retrain on full training data to generate predictions for the test data.
