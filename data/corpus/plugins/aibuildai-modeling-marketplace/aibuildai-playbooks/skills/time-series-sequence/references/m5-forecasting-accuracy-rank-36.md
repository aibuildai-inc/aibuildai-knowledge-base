# [36th place] New forecasting Package: tsboost

Competition: m5-forecasting-accuracy
Rank: #36
Source: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163346

First, Congrats to all winners and participants !

I would like to introduce a new open source package I created few time ago for forecasting time series with discrete time step, installable through pypi and on my github :

https://github.com/franck-durand/tsboost

The goal of this framework is to provide a new tool ready to go in production for data scientists (like prophet), so that they can prototyping faster.

I will publish my full solution for M5 on my github as it uses this package plus some data pre processing &amp; some feature engineering. But as I am a generally lazy personn, I will do it... in a more or less near futur.

I did also 15th place for M5 uncertainty, and I plan to program the methodology I used in this time series framework in "a near future". By this way it will be possible via a meta parameter of the package to generate confidence intervals along with forecast point.

It was a really interesting double competition. And I really like the testing methodology for time series : 
- 100% of the score is from private LB data (No bias from public Leader board in final evaluation)
- that we can submit only one submission (it's like you are in a production context for time series)
- unfortunately it was still possible to have a bias with futur external datas not avaiable at the time the forecast was supposed to be done (for exemple US GDP of june 2016, not available at the end of may 2016)
- maybe the metric was not an easy one, but still, I belive it's a good one for hierachical time serie problems.

I'm also a little bit sad for @kyakovlev , I belive he has taught a lot of good things and ideas to kagglers, and he really deserved to be at least in the gold zone.

Regards !
