# 10th place solution

Competition: santander-customer-transaction-prediction
Rank: #10
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88997#latest-517584

Thank kaggle &amp; santander for this interesting competition, my solution is quite straightforward:

1.  single model of NN with  input shape (200, 2) and in class shuffle augmentation (similar as explained in  [5th place solution](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88929#latest-513419)) reach 0.9239/0.9232 on public/private LB.
2. The final solution of 0.92502/0.92421 LB is a simple ensemble of a few NN models trained with different ratios of positive/negative in data augmentation.
3. I believe some extra lightgbm models would significantly help in ensemble due to low correlations with NN models.  Unfortunately I am not familiar with the methodology and my lightgbm model never break 0.922.

Thank everyone for sharing on the forum, I have learned/enjoyed a lot reading them.
