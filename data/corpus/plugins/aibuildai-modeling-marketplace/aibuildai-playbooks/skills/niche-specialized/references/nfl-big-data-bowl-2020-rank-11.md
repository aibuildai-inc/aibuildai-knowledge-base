# Public 17th place overview

Competition: nfl-big-data-bowl-2020
Rank: #11
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119330

[Data standardization]

```
df['S'] = df['Dis']*10
```

Host says that:

```
Dis measures time covered in the most recent window of player tracking data.
Given that tracking data roughly covers 10 frames per second,
Dis corresponds to distance traveled in the recent 0.1 seconds.
Note that speed and acceleration are directly calculated using Dis (this is done in the data pre-processing)
```

[Features]
I created about 300 features.
・N seconds later minimum distance between rusher and other players
・N seconds later minimum (distance/S) between rusher and other players
・N seconds later how many players around rusher(square, circle)
...etc

```
# N seconds later player's position
df['X_after_N_seconds'] = df['X'] + df['S'] * np.cos(np.deg2rad(df['Dir'])) * N
df['Y_after_N_seconds'] = df['Y'] + df['S'] * np.sin(np.deg2rad(df['Dir'])) * N
```

[Modeling]
・5 fold with stratified kfold
・Binning target and treat as classification

・NN ... CV 0.01234
Structure is here.

```
x = Concatenate(axis=1)([category_features_embedding, num_features])
x = Dense(150, activation='softplus')(x)
x = Dropout(0.5)(x)
x = Dense(100, activation='softplus')(x)
x = Dropout(0.25)(x)
predictions = Dense(num_classes, activation='softmax')(x)  
```

・Lightgbm ... CV 0.0125
Use conservative parameters.
ex 'num_leaves': 10, 'max_depth': 3, 'min_data_in_leaf': 150, 'max_bin': 64

[Postprocess]
This give +0.00005 on LB.
For example, if yardline is 30, possible gain yards within -30 to 70

```
max_yard = 99 + 70
min_yard = 99 + -30

pred[max_yard-1] += pred[max_yard:].sum()
pred[max_yard:] = 0

pred[min_yard+1] += pred[:min_yard].sum()
pred[:min_yard] = 0
```

[Final submission]
Averaging 2 NN and 1 lightgbm.

[Doesn't work]
・Predicting outliers
I think predicting long Yards exactly will give big jump.
But I couldn't succeed it.

・Using A of season 2017
This cause overfitting with me.
How can I deal with A of 2017...?
