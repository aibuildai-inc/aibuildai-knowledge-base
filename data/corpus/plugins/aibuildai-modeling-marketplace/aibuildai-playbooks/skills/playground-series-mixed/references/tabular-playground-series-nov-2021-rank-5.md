# #5 solution (and +141 from Public Leaderboard)

Competition: tabular-playground-series-nov-2021
Rank: #5
Source: https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291846

1. I found that MaxAbsScaler transform data better than Robust / Standart or Power (Gauss)

2. Some trigonometric functions slightly improved the result, like sin(f27)

3. I used Keras NN with 4 Dense layers with high amount of units (starting from 600) with high dropout (0.3), swish activation (better than relu)

4. It was important to choose the correct learning rate and learning rate scheduler for long epoch training and pretty curves

5. When the situation became clear with chunks I started to use simple KFold(shuffle=False) with 10 splits - for using exactly one chunk to validate and nine to learn.

6. SVM postprocessing some improved the result (thanks @ambrosm)

7. When it became clear that the top was strongly overfitted, just waiting for the end of the competition :)
