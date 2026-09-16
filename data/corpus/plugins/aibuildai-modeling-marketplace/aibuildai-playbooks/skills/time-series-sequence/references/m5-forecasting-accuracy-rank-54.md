# 54th place solution [No lottery - just a baseline]

Competition: m5-forecasting-accuracy
Rank: #54
Source: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163183

First of all I want to thank all Kaggle community: besides you can learn lots of new stuff here almost on every competition, community is very welcomed. 

In this post I want to share my approach, for me the result was not surprising.

When the competition was firstly announced I was  really motivated to invest lots of time into it in order to get a high place, but I faced some challenges at my work, so my competition-start was postponed and I started working under the competition about 1 month before the deadline when public LB had been already published.

Based on my CV and lots of top-scoring public notebooks with custom multipliers I decided to build just a good baseline. 

**Validation**
In my opinion it was the most important part.

I validate my score on 3 folds:

- (2016-03-27;2016-04-24]
- (2016-04-24;2016-05-22]
- (2015-05-17;2015-06-14]

In my 3d validation fold I tried to choose aprox the same period of year - the same as final evaluation part.

**Pre-processing**
There were some outliers and spikes in each individual ts so I did some smoothing:
using MAD (median absolute deviation) of last 180 days of an item to find outliers and replace it with value 7 days ago - without this smoothing rolling mean features which included holidays (like Christmas) didt work in proper way.

**Models**

- Standart LGBM with Tweedie Loss (1 model for all dates)
`params = {
                'boosting_type': 'gbdt',
                'objective': 'tweedie',
                'tweedie_variance_power': 1.1,
                'metric': 'rmse',
                'subsample': 0.75,
                'subsample_freq': 1,
                'learning_rate': 0.03,
                'num_leaves': 2**11-1,
                'min_data_in_leaf': 2**12-1,
                'feature_fraction': 0.7,
                'max_bin': 100,
                'n_estimators': 1400,
                'boost_from_average': False,
                'verbose': -1,
            } `

- Wavenet with GRU-part

Final submission was weightened average with 0.8 and 0.2 coefs for models.

**Target**
For my LGBM I didnt use **Demand ** as a target  - instead I used ***Difference*** of Demand and Demand of 28 days ago - for tree-based model it`s a nice trick to work wih trends. 

**Post-processing**
No custom coefficients :)

**Features**
Pretty standart, most of them can find in public notebooks. 
In my CV worked pretty good ratios of rolling mean and median.

`data['ratio_mean_60'] = data['rolling_mean_t60'] / data['rolling_median_t60']`

**What didn`t work**
- custom loss with WRMSSE gradient 
- out of stock prediction (spent lots of time on it - didnt way right way)
-  mean encodings
- hts-method with upper-level correction
- and lots of other ideas from previous competitions and public ideas


It was a really great journey working under this competition - learned lots of tricks as usual, somehow this competition reminded me a LANL competition with a huge shake-up and custom multipliers. 

Big congratulations to winners and big thanks to @kyakovlev, @girmdshinsei, @sibmike which notebooks and ideas i included in my solution.
