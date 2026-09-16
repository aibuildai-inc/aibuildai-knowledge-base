# 4th place solution

Competition: cat-in-the-dat
Rank: #4
Source: https://www.kaggle.com/c/cat-in-the-dat/discussion/121584

Logistic regression  with the following encodings of the columns:

1. binary: linear transform to [-1, 1]
2. nominal: collapse unique values from train/test to single value, one hot encode to sparse matrix (both this ideas saw in public kernels)
3. ordinal: transform to [-1, 1] using sklearn.preprocessing.MinMaxScaler((-1, 1))
4. cyclical: one-hot encode. regression coefficients for day looked symmetrical, so tried to one hot encode abs(day-4) instead of day. 

The optimizer was sklearn.linear_model.LogisticRegression(solver='lbfgs', C=.121)

Most of the time went into trying different encodings for day and month, hoping to find cyclical nature of this columns.
