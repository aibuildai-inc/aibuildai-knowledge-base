# 19th place solution

Competition: liverpool-ion-switching
Rank: #19
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153721

Single 1cnn wavenet with batchnormalization and a little bit of dropout with the clean kalman dataset of my public wavenet notebook.

Important:
Added some data augmentation to handle the shift in the test adding a constant to the signal.
Another data augmentation technique that also improve the cv was to add a little bit of a random normal distribution like this train_aug['signal'] + [np.random.normal(0, 0.01) for x in range(train_aug.shape[0])]. What i actually used was train_aug['signal'] + np.random.normal(0, 0.01) hoping that this will handle the shift and it worked (this was pure luck :) )

Used KMeans to create 22 clusters, then i one hot encode this cluster making 22 extra features.

Also i get some aggregated stats for each cluster like the kurtosis, skewness and std.

Added the difference between the signal and nearest cluster centroid, also added the euclidean distance between them

Used -+ 3 shifted signal, -3 rolling mean and std, -1 to 1 rolling mean and std, signal ** 2 and signal ** 4.

That cover the feature engineering part. 

To handdle overfitting i used a learning rate scheduler like the public wavenet so it can make an early stop (if i used constant lr of 0.001 the net overfitts badly giving 0.942 public and private leaderboard and an awsome oof f1 of 0.94452)

It was a fun competition, msg me if you want to team up in another competition that we are both interested and have more or less the same skills and knowledge

Sry for my english it's not my first language.
