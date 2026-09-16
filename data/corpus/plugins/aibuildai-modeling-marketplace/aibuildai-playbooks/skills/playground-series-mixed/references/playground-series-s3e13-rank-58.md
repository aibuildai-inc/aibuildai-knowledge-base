# #58 solution: CV = 0.475, Public LB = 0.43, Private LB = 0.492

Competition: playground-series-s3e13
Rank: #58
Source: https://www.kaggle.com/c/playground-series-s3e13/discussion/406396

Hello everyone, I would like to take a moment to express my gratitude for the valuable codes and discussions shared during the competition. I would like to briefly outline my solution in a few steps:
1. I chose not to use the original dataset in my solution.
2. I calculated the frequency of each feature across all 11 diseases.
3. I then computed the **z-score** of these frequencies for all the features.

###The reason I used z-score:
I chose to use the z-score because it allows for the relative importance of each feature to be weighted based on its frequency in each disease. For example, the presentation of **toenail_loss** is in the power of **2.78 for Tungiasis**, but in contrast, it is **-0.67 for Chikungunya**. This gives more credit to the presentation of important features for their respective diseases and also gives a negative value to them for other diseases.

4_ I clustered these symptoms based on the z-scores into three clusters using **KMeans** and then summed up the values (0 or 1) for each cluster to add **3 new features** to the df_train dataset.


5_ I summed up the z-score values for each cluster of symptoms for all 11 diseases(presenting ones). Therefore, for each of the three clusters, I computed the scores of the 11 diseases, resulting in the addition of **33 new features** to the dataset.

Having these 36 features and dropping all of the original features, I achieved the following results:
```
CV = 0.475
Public LB = 0.43, Rank = 41
Private LB = 0.492, Rank = 58
```
