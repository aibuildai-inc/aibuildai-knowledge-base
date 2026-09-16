# 1st place solution

Competition: bosch-production-line-performance
Rank: #1
Source: https://www.kaggle.com/c/bosch-production-line-performance/discussion/25434

Sorry for the delay. It was good to have a kaggle free weekend after two months. (Ok not completely kaggle free but at least in read only mode:).

As we had 6-7 hours local time difference (USA, EU) we tried to split the work. 
I was mostly doing feature engineering and tried to improve our best single xgb score. Ash tried all kinds of different modeling techniques and built the L2/L3 ensemble.

## Data Exploration
I spent the first two weeks with data discovery. Numeric feature plots, statistics, station to station transition probabilities, correlation matrices for each stations was necessary to have ideas how to handle the thousands of anonymous features.

I start each competition with desktop machine (16GB RAM). It takes some extra effort in the beginning to deal with lower level data manipulations due to the memory constraints. Getting closer to the raw data helps sometimes later. 

As the numeric/date features had .3f/.2f precision we could rescale (*=1000, *=100) everything to integers. We kept only the min time for each station.
Sparse matrices came handy as we had a lot of missing values. 


## Leak/Magic/Data Property Features
Ash found quite early that consecutive rows had feature duplication and correlated Response. We called consecutive Ids ordered by StartStation, StartTime *chunk*s.

For each chunk we used features

* number of records
* rank ascending
* rank descending

We used the original order to add features based on previous and next records.

* Response
* feature hash equality for each station
* StartTime, EndTime
* Numeric raw features

## Time Features
I figured out in the beginning that [0.01 time granularity means probably 6 mins][1].

So we had to deal with 2 years of manufacturing data. This observation did not gave direct performance boost but gave us enough intuition to construct a lot of time based features.

* StartStationTimes
* StartTime, EndTime, Duration
* StationTimeDiff
* Start/End part of week (mod 1680)
* Number of records in next/last 2.5h, 24h, 168h for each station
* Number of records in the same time (6 mins)
* MeanTimeDiff since last 1/5/10 failure(s)
* MeanTimeDiff till next 1/5/10 failure(s)


## Numeric Features
* Raw numeric features (most of the time we used the raw numeric features or simple subsets based on xgb feature importance)
* Z-scaled features for each week
* Count encoding for each value 
* Feature combinations (f1 + -  * f2)

We saw interesting similarity in the station transition probability matrix between station S0-S11 and S12 - S23. You have probably seen the same in the Shopfloor visualizations. We also noticed that the number of features are the same across these station groups.

![enter image description here][2]

The correlation matrices were similar for these stations so we tried combining the numeric features for the same stages. (e.g. L0_S0_F0 + L0_S12_F330 etc.)



## Categorical Features
We could not squeeze much out of the categorical features. Most of the time we just dropped them. We had a few models with aggregated categorical features for each station in [24, 25, 26, 27, 28, 29, 32, 44, 47]. 'T1' was replaced by 1 all the other values were replaced 100.

## Validation
Given the imbalanced labels and binary evaluation criteria we noticed quite high 5-FOLD CV stds (~0.01) in the beginning. After a few experiments we decided to use 4 FOLD "leak-stratified" CV (forcing to have the same number of duplication for each fold). 
Each "single" L1 model was trained with 3 different seeds on the same folds. Our latest ensemble scores were quite stable (std < 0.004) and very often our CV/LB score improved hand in hand.

## Rebalancing
We kept every failure and records with duplication. For the remaining 0s 50-90% down sampling was used. It made the training a bit faster and worked as bagging for the model averaging part helping the later ensemble stages.

## Results
In the last month of the competition we continuously improved and tried to keep us in the top 3. Fortunately we could improve significantly both CV and LB  with our last submissions on Friday. This made the submission selection easier.
![enter image description here][3]


  [1]: https://www.kaggle.com/gaborfodor/bosch-production-line-performance/notebookd19d11e4f2
  [2]: https://kaggle2.blob.core.windows.net/forum-message-attachments/144544/5351/StationFeaturSimilarity.png?sv=2012-02-12&se=2016-11-17T16%3A25%3A47Z&sr=b&sp=r&sig=pEgr%2B4FRBrS2Owl2ctg5sOEMBzMbWQoL78R%2FEed8h%2BA%3D
  [3]: https://kaggle2.blob.core.windows.net/forum-message-attachments/144544/5348/results.PNG?sv=2012-02-12&se=2016-11-17T16%3A01%3A24Z&sr=b&sp=r&sig=%2FBvQoJg5kEuzvBWFBoHFs7GU%2Fgr7AotglS54yYgZk3M%3D
