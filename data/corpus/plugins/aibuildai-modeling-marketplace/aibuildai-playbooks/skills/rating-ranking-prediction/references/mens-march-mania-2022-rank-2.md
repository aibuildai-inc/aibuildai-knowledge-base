# #2 Solution

Competition: mens-march-mania-2022
Rank: #2
Source: https://www.kaggle.com/c/mens-march-mania-2022/discussion/318779

Competition Name: March Machine Learning Mania 2022 – Men’s
Team Name: shin koshi
Private Leaderboard Score: 0.55415
Private Leaderboard Place: 2

1. Background
・An undergraduate student at the faculty of information and computer science, Chiba Institute of Technology.
・This is my first time participating in Kaggle's MLMM competition.
・I spent two weeks on this competition.

2. Summary
My #2 place solution code is [here](https://www.kaggle.com/code/shinkoshi/2nd-place-solution).
I used LGBM as a model. The model was implemented using AutoLGBM. It takes two hours to train.

3. Features
I used only some of the datasets distributed in this competition.
The following features were used.
・MNCAATourneyDetailedResults.csv
・MNCAATourneySeeds.csv
・MMasseyOrdinals_thruDay128.csv

4. Model
・Training LGBM with AutoLGBM library.
・I trained with 10-fold cross-validation.
・The notebook was forked from @PIXYZ0130 [notebook](https://www.kaggle.com/code/pixyz0130/eng-mmlm-men-autolgbm-baseline), and most of the places were just copied. The grouping and other small details are slightly different.
