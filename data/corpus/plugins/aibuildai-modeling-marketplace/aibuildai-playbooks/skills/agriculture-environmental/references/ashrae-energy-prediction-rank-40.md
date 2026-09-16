# 40th place Public solution. My part.

Competition: ashrae-energy-prediction
Rank: #40
Source: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/122863

As we are still waiting for the private leaderboard I have decided to share our solution that led us to the 40th place at public LB.
Enjoy the reading, good luck with a shakeup and a happy new year to everyone.

# Data cleaning
The best score boost in this competition was achieved thanks to data cleaning. There are plenty of training examples that might be removed. For example building_id's from 0 to 104 have all 0's (or almost 0's) for the period starting from January 1st 2016 until May 20th 2016. So all of this samples should be removed from the training set.


Some buildings have suspicious 0 readings somewhere in the middle, like building_id 1066

Code to remove data:
`"not (building_id == 1066 &amp; meter == 0 &amp; timestamp &gt;= "2016-02-13 20:00:00" &amp; timestamp &lt;= "2016-02-29 11:00:00")"`

And another example - building_id 1250 &amp; meter 2. It has all 0's until the middle of December. So my guess was that this type of meter was not functioning, but started at December. Based on that guess the small amount of non-zero data is very useful, especially because the readings are so high (around 8000).

Code to remove data:
`"not (building_id == 1250 &amp; meter == 2 &amp; timestamp &lt; "2016-12-21 16:00:00")"`

I have been using my plots from [this post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/115248) to visually understand which readings might be removed.

Overall our file with data cleaning queries contains 102 lanes. That is including both training set and leaked set. Yes, we did cleaned leaked data and used it as a training examples.

# CV strategy
2 Folds strategy from [Half and half](https://www.kaggle.com/rohanrao/ashrae-half-and-half) showed a really good performance. But since it splits all data simply in the middle it has disadvantages because some of the buildings have samples only starting from June, July or even November. So they will be only presented in one fold.

So I have created a slightly different strategy - using half and half, but now for the whole training set, but for every meter of every building in it. In other words - if building have readings only for 6 months, starting from July, then 1st fold will contain all the readings from July till September and 2nd fold will contain readings from September till December.

Below is a picture with an example. Pay attention that data for building_id 555 starts only from July.


And here is the code to create two folds for the whole dataset, splitting each meter of each buildings somewhere in the middle:

```
fold1 = []
fold2 = []
folds = [fold1, fold2]

# Iterating over each building
for building in tqdm_notebook(range(1449)):
    # And over each possible meter type for each building
    for meter in range(4):
        sub_df = train[(train['building_id']==building) &amp; (train['meter']==meter)]
        # Dividing by two, because we will train on 2 folds
        samples = int(sub_df.shape[0] / 2)
        # If there are any samples for the specific meter type of the specific building...
        if samples:
            # Filling indexes for both folds. 
            # They will be splitted by time, since at this moment train is sorted by time.
            fold1 += list(sub_df.index[:samples])
            fold2 += list(sub_df.index[samples:])
```

This CV strategy performs better and OOF score better aligns with LB.

# Feature engineering
We have used around 50 features in our model. Some of the features are well known from public kernels, such as lags. But some of them were mined using CV strategy.

One of the features is a temperature difference between two readings for each building_id. It improved a score a bit.
Code example:
`train['air_temperature_diff1'] = train['air_temperature'] - train.groupby(['building_id', 'meter'])['air_temperature'].shift(1)`

Also we have used holidays as a features, SIN\COS transformations and so on.

# Model
We mostly used LightGBM as it showed best performance. I believe we used only one CatBoost model for the blending. Neural Network didn't show a good performance.
Final submission is a blend of around 10 models (or some number around that).

# Leaked data
Yes, we used leaked data both for training and in the submission file to replace our predictions with ground truth labels.
