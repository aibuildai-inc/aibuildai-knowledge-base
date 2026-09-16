# 40th place solution, my part.

Competition: ieee-fraud-detection
Rank: #40
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111245

### Preamble

First of all I want to congratulate all the winners, as you will get paid for your hard work. Also personal congratulation to @cdeotte with his 'promotion' to competitions master. 3 more gold medals and we will have one more triple GM. 
Congratulations to @fatihozturk for becoming competitions GM.
And congratulations to all other participants because the best prize you are getting out of every competition is what you have learned.

### Solution

- We did find the same thing as, I suppose, any other team from top 1-2% has found, which is uid by card1, addr1, email and TransactionDT - D1. 
- So then comes a lot of different groupby's on that uid
- Also my teammate @nvarganov has engineered some fancy shifting features, which added around 0.0005 to my best single model when I used them
- All in all my best model had ~1000 features. It was a LightGBM on 10 KFolds with no shuffle and it scored 0.956 Public and 0.932 Private
- Our best submit was a blend of all of our best models, which contained LightGBM and XGBoost.

### C features interaction
And also here is a code for some C features interaction which I used in my best model and which improved it's score by a little bit. I believe it was around 0.0002-0.0003.

A list of C features
```
c_cols = [col for col in X.columns if col.startswith('C') and len(col) &lt;= 3]
```
Multiplication and subtraction interactions each-to-each
```
for i in range(len(c_cols)):
    for j in range(i + 1, len(c_cols)):
        X['{}_mul_{}'.format(c_cols[i], c_cols[j])] = X[c_cols[i]] * X[c_cols[j]]
        X['{}_sub1_{}'.format(c_cols[i], c_cols[j])] = X[c_cols[i]] - X[c_cols[j]]
        X['{}_sub2_{}'.format(c_cols[j], c_cols[i])] = X[c_cols[j]] - X[c_cols[i]]
        test['{}_mul_{}'.format(c_cols[i], c_cols[j])] = test[c_cols[i]] * test[c_cols[j]]
        test['{}_sub1_{}'.format(c_cols[i], c_cols[j])] = test[c_cols[i]] - test[c_cols[j]]
        test['{}_sub2_{}'.format(c_cols[j], c_cols[i])] = test[c_cols[j]] - test[c_cols[i]]
```

Computing a correlation matrix for all new generated features
```
sum_cols = [col for col in X.columns if '_mul_' in col or '_sub_' in col]
corr_matrix = X[sum_cols].corr()
```

Then removing all the features with a high correlation. Keeping those which correlate with target value better.
```
to_drop = list()

# Iterating over rows starting from the second one, because position [0, 0] will be self-correlation which is 1
for i in range(1, len(corr_matrix)):
    # Iterating over columns of the row. Only going under the diagonal.
    for j in range(i):
        # See if the correlation between two features are more than a selected threshold
        if corr_matrix.iloc[i, j] &gt;= 0.98:
            # Then keep the one from thos two which correlates with target better
            if abs(pd.concat([X[corr_matrix.index[i]], y], axis=1).corr().iloc[0][1]) &gt; abs(pd.concat([X[corr_matrix.columns[j]], y], axis=1).corr().iloc[0][1]):
                to_drop.append(corr_matrix.columns[j])
            else:
                to_drop.append(corr_matrix.index[i])

to_drop = list(set(to_drop))
```
