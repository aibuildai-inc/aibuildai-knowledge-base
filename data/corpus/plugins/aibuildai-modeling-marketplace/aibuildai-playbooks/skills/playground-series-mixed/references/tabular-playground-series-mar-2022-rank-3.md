# 3rd place, 3rd month, 3 submissions

Competition: tabular-playground-series-mar-2022
Rank: #3
Source: https://www.kaggle.com/c/tabular-playground-series-mar-2022/discussion/317661

Hi, 

I surprisingly made the third place by jumping more than 300 places. But if you check out the top 20 in the private leader board, you’ll notice all did similar jumps. So I guess I was quite lucky. 

At any rate, my approach:
- It’s a timeseries problem:
  - --> use [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) to evaluate model performance
  - Time features: day of week, is Monday, hour of day, minute of hour
  - Lag the target by 7 and 14 days
- Model = slightly tuned [fastai tabular learner](https://docs.fast.ai/tabular.learner.html):
  - Number of epochs: 8
  - Learning  rate: 0.005677
  - Layers: [1066, 931]
  - Batch size: 256


Have a wonderful day! :)
