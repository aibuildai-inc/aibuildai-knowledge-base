# 20th place solution based on custom sample_weight and data augmentation

Competition: recruit-restaurant-visitor-forecasting
Rank: #20
Source: https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/49328

Thank you for my teammate Taka, Kaggle Admin, Recruit company, and all participants of this exciting competition. We would like to share our 20th solution, which is based on target encoding, customized sample\_weight, data imputation and augmentation.

## 1. Problem setting
We were required to predict Japanese restaurants visitors from the restaurants' attribute information and historical information. Although the training data covers the dates from 2016 until April 2017, most of the stores are not observed January 2016. In terrible case, its first observation is March 2017, which is just before the test period.

## 2. Validation method
We split the last one month of the train data as validation set before conducting any feature engineering. The validation set was split to first 6 days as pseudo-Public and the rest as pseudo-Private. This method was really important for avoiding overfitting to Public Leaderboard and to know genuine performance of our models.

## 3. Feature engineering
We mainly conducted two feature engineering methods. One is target encodings and the other is creation of weather-based features. In addition, we used some ideas from public kernel carefully.

### a. Target encodings
We conducted target encodings with combinations of "attribute information of the store" and "time-based features". Various statistics were created by this encoding as below.

- attribute information of the store
air\_store\_id, air\_genre\_name, air\_area\_name, prefecture

- time-based features
year, month, day, day\_of\_week, is\_holiday (including Sat and Sun), is\_yesterday\_holiday, is\_tomorrow\_holiday, rolling mean of is\_holiday (forward), rolling mean of is\_holiday (backward) etc.

- statistics
mean, min, max, sum, var

### b. Weather features
- Daily weather (precipitation, daylight, temperature)
- Inner product of "hourly weather vector" and "hourly visitor vector".
- Getting sick index, which is distance from optimal temperature and humidity for virus. We conducted 7 day rolling mean of the index considering virus latency period. This feature was more effective than we expected.

## 4. Sample weight
We designed the sample weight as the sum of two sample\_weights (a + 0.3*b) considering the importance of recent events and annual periodicity. All parameters such as half-life was determined experimentally by validation scores. For validation and test, we used sample_weights with different values produced by the same logic, rather than sample\_weights with same values. 

### a. Exponentially decayed weight (sample\_weight\_a)
We created an exponentially decayed weight with half-life of 60 days.

### b. similarity-based weight (sample\_weight\_b)
We defined similarity of each months to the target period as predictability by LightGBM.

![sample_weight][1]
Fig 1. Customized sample_weight designed as sum of two components

## 5. Data imputation and augmentation
### a. Imputation
Although month-based engineered features were really effective for public and pseudo-public score, it caused a catastrophic result to the pseudo-private score. Therefore, we imputed missing values of these statistics of May by ensemble of various regressors. We used the attribute features of the stores and the trend of 2016 as training data and these of 2017 as test data for this imputation.

### b. Augmentation
Our hypothesis is that the first observation date of the stores is simply date of registration and not date of grand opening. Therefore, it is arbitrary decision by store owner and it should not change the visitors substantially. Based on this hypothesis, we randomly dropped past observation of both of train and test data. After that, obtained predictions were averaged. We expected that this processing increases the number of training data and make our model more robust. It should be noted that this processing is different from simple subsample because target encoding is conducted after augmentation and it brings change to created features.

![drop_out past obs][2]
Fig 2. Stochastic drop of past observations as data augmentation

## 6. Result
20th/2158.
We couldn't get gold medals! Will try next time again!


  [1]: https://storage.googleapis.com/kaggle-forum-message-attachments/280160/8536/custom_weight.jpg
  [2]: https://storage.googleapis.com/kaggle-forum-message-attachments/280160/8537/drop_out.jpg
