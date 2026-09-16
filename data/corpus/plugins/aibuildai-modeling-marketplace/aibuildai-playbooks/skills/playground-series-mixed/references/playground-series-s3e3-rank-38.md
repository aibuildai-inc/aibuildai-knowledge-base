# My S3E3  226 - >> 38

Competition: playground-series-s3e3
Rank: #38
Source: https://www.kaggle.com/c/playground-series-s3e3/discussion/380748

This episode frustrated me a lot since I couldn't improve my public leaderboard score from day 2, but it ended up giving me a good result in the private leaderboard. I just cleaned up the data and gave only one new feature "MonthlyIncome/Age" as I posted on the [discussion](https://www.kaggle.com/competitions/playground-series-s3e3/discussion/379076). 

My model was very simple. I used the competition and original data for training.

 ```python
model = CatBoostClassifier(verbose=0,n_estimators=500)
predictions_cat = make_predictions(full,5,model)
cat =[np.mean(a) for a in zip(*predictions_cat)]
```
I have a note on how I developed my notebook at the end.  If you are interested, please have a look at [my notebook](https://www.kaggle.com/code/satoshiss/employee-attrition-prediction-ps3e3). The best score was from version 8 of this notebook.

Hope this helps someone here, and see you soon in the next episode.👍
