# 30th Place Solution

Competition: nfl-big-data-bowl-2020
Rank: #30
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119301

### 1 - Features

First of all, we used the 62 features below. You can understand what they mean by their names:

- 'Yards_to_touchdown',
- 'YardLine_ref',
- 'X_defense_spread',
- 'X_defense_std',
- 'Y_defense_spread',
- 'Y_defense_std',
- 'X_offense_spread',
- 'X_offense_std',
- 'X_offense_centroid',
- 'X_defense_centroid',
- 'DistToBallCarrier_Offense_Mean',
- 'DistToBallCarrier_Defense_Mean',
- 'DistToBallCarrier_Offense_std',
- 'DistToBallCarrier_Defense_std',
- 'DistToBallCarrier0.5_Offense_Mean',
- 'DistToBallCarrier0.5_Defense_Mean',
- 'DistToBallCarrier0.5_Offense_std',
- 'DistToBallCarrier0.5_Defense_std',
- 'Y_offense_spread',
- 'Y_offense_std',
- 'Y_offense_centroid',
- 'Y_defense_centroid',
- 'Seconds_since_quarter',
- 'Distance_to_down',
- 'DefendersInTheBox',
- 'Average_tackle_time',
- 'Min_tackle_time',
- 'Season_cat_2017',
- 'Season_cat_2019',
- 'A1',
- 'S1',
- 'X_estimated1',
- 'Y_estimated1',
- 'Dir_cos1',
- 'Dir_sin1',
- 'S1_vs_S12',
- 'A12',
- 'S12',
- 'X_ref12',
- 'Y_ref12',
- 'X_estimated12',
- 'Y_estimated12',
- 'DistToBallCarrier12',
- 'DistToBallCarrier0.5_estimated12',
- 'Dir_cos12',
- 'Dir_sin12',
- 'Min_tackle_time12',
- 'S_horizontal12',
- 'S_vertical12',
- 'A13',
- 'S13',
- 'X_ref13',
- 'Y_ref13',
- 'X_estimated13',
- 'Y_estimated13',
- 'DistToBallCarrier13',
- 'DistToBallCarrier0.5_estimated13',
- 'Dir_cos13',
- 'Dir_sin13',
- 'Min_tackle_time13',
- 'S_horizontal13',
- 'S_vertical13'

A key point to improve our score was to sort the dataframe by using this line of code:
```
train = train.sort_values(by= ["PlayId", "IsOnOffense", "IsBallCarrier", "DistToBallCarrier0.7_estimated"], ascending=[1, 0, 0, 1])
```
The “DistToBallCarrier0.7_estimated” feature is the estimated distance of a player to the rusher after 0.7 seconds.

### 2 - Cross Validation
We tried RepeatedKFold and GroupKFold with “GameId”, “Week” and “NflIdRusher” separating the groups. The RepeatedKFold strategy gave a better result:
```
rkf = RepeatedKFold(n_splits=3, n_repeats=4, random_state=1301)
```
### 3 - Training
In order to train our model we used Keras with a very simple neural network:
```
model = keras.models.Sequential([
        keras.layers.Dense(units=256, input_shape=[X.shape[1]]),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.4),
        
        keras.layers.Dense(units=256),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(units=256),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.2),
    
        keras.layers.Dense(units=199, activation='sigmoid')
    ])
```
### 4 - Post Process
For post processing we used these three lines of code:
```
y_pred[:, :50] = 0
y_pred[:, -yards_covered:] = 1
y_pred[:, :100-yards_covered] = 0
```
Intuition:
- The 50 first predictions is 0 because there isn't any team which lost a huge amount of yards on the training set.
- The two last lines of code mean that if the yardline is, let’s say 75, the maximum number of yards the team can gain is 25. So every prediction on the right side of +25 should be 1. The same applyes for the case of lossing yards, where the team can losse only 75 yards and every prediction on the left side of -75 should be 0.

### 5 - Score
CV score: 0.01213
Public LB: 0.01297
Private LB: 0.012669

**Thanks for reading 👍 and many thanks to my teammates @jayjay75, @rafiko1 and @bolkonsky!**
