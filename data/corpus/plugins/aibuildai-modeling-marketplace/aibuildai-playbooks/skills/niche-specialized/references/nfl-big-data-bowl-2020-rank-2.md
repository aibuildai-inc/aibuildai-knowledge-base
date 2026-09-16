# 2nd Private LB/ 3rd Public LB Solution - Transformer

Competition: nfl-big-data-bowl-2020
Rank: #2
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119314

# **Data Cleaning**
As we all know data in 2017 is different from 2018, data cleaning is very important in this competition.
- Orientation: 90 degree rotation in 2017
- A: I cannot find a good way to standardize A, I replace A in 2017 by the mean, surprisingly this improve my LB by 0.0002
- S: if we look at 2018 data, we can see that S is linearly related to Dis

While data in 2017 is not very fit,

By fitting a linear regression on 2018 data, the coefficient of lr is 9.92279507, which is very close to 10, so finally I replace S by 10 * Dis for both 2017 and 2018 data. This also gave me 0.0002 improvement.

# **Features**
total 36 features, ['IsRusher','IsRusherTeam','X','Y','Dir_X','Dir_Y',
            'Orientation_X','Orientation_Y','S','DistanceToBall',
            'BallDistanceX','BallDistanceY','BallAngleX','BallAngleY',
            'related_horizontal_v','related_vertical_v',
            'related_horizontal_A','related_vertical_A',
            'TeamDistance','EnermyTeamDistance',
            'TeamXstd','EnermyXstd',
            'EnermyYstd','TeamYstd',
            'DistanceToBallRank','DistanceToBallRank_AttTeam','DistanceToBallRank_DefTeam',
            'YardLine','NextX','NextY',
            'NextDistanceToBall',
            'BallNextAngleX','BallNextAngleY',
            'BallNextDistanceX','BallNextDistanceY','A']

# **Cross Validation**
Always include 2017 data for training, 3 group folds by week for 2018 data, use only 2018 data for evaluation.  In this way the CV score is close to public LB.

# **Model**
Transformer (2 layers encoder + 2 layers decoder), large number of attention head is the key
.png?generation=1574922346398274&amp;alt=media)

# **Model ensemble**
Optimizer: RAdam + lookahead
Number of epoch: 30
Batch Size: 32
Weight Decay: 0.1
Ensemble: snapshot ensemble (pick models at epoch 11, 13,...,17,29)
Learning rate scheduler: 8e-4 for epoch 0-10,12,14,...,28. 4e-4 for epoch 11,13,...,29

Since we are only given 4 hours CPU training, snapshot ensemble seems to be a perfect choice as it won’t increase our training time and is significantly better than single model.
In my final submission, I repeat the training (use all data) for 11000s and 9000s (safe mode).
