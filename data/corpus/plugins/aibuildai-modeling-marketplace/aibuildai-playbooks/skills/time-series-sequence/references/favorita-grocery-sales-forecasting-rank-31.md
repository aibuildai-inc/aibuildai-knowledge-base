# A silver solution(33rd)

Competition: favorita-grocery-sales-forecasting
Rank: #31
Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47537

I'm a newbie in Kaggle and it's actually a shock for me to reach top 2%...

My solution is based on Lingzhi's LGBM One Step Ahead with some modifications:

- using 8 weeks of training data starting from May 31st, 2017

- extract min, max, var, skew and kurt

- calculate first order difference for the mean values

- Categorical features(store, item): one hot encoding, count (I learn this trick from https://www.kaggle.com/xiaozhouwang/2nd-place-lightgbm-solution)

- averaged across different random seeds to stabilize model performance (also learned from Little Boat's solution)

I have posted my LGBM model on https://www.kaggle.com/deepdreamer/a-silver-solution-33rd. It scores 0.517 in private board.

Thanks Little Boat for sharing his solutions in Porto Seguro and Ceshine lee for sharing the framework. I really learn a lot from them.
