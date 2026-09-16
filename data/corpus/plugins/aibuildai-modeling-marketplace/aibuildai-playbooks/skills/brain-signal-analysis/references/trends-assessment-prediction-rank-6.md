# 6th place solution

Competition: trends-assessment-prediction
Rank: #6
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162828

**Amazing team for professional growth, victories and interesting communication:** Sergey Bryansky @sggpls, Andrei Lupu @andreils, Möbius @arashnic, Yury Bolkonsky @bolkonsky and Kirill Shipitsyn @kirillshipitsyn  👍 😃 

**Thanks all participants and organazers for such interesting competition!**

**FE:** we converted 4d images to 3d and train 3D CNN Autoencoder for each spatial map (1,1008*53) for each case, using PCA with different kernels to compress the data to (1,2048) for each.

We used these features + original + site classifier features for training our pipeline. 



Also we were using target binning technique and feature selection which improve our final score **(0.15663 public LB 6th place, 0.15703 private LB 6th place).** 🔥
