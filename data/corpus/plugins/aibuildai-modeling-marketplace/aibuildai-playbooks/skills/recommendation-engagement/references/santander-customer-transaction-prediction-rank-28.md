# 29th place solution with code

Competition: santander-customer-transaction-prediction
Rank: #28
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89034#latest-519546

#### Code and more detailed writeup is on github: https://github.com/btrotta/kaggle-santander-2019

My solution is quite short, &lt;200 lines of code. Training and prediction takes around 3.5 hours. 

I relied heavily on some excellent kernels and discussions by others, which I reference below. 


### The magic

There are 3 key observations, all related.

*Feature independence:*
It was discovered in some excellent early kernels that the features seem to be independent given the target (see kernel and discussion by Branden Murray (1), (2) and kernel by Chris Deotte (4). So it appears likely that each feature column has been independently "scrambled" within each target class (0 or 1). This means there's no point looking for feature interactions among the columns.

*Synthetic test data:*
The kernel by YaG320 (3)) shows that some of the test data contains no unique feature values in the whole row, suggesting that it has been synthesised from the test of the data.

*Repeated values:*
Repeated values seem to show a stronger signal, as shown in Figure 1.1. I wasn't able to figure out why this is, nor whether it's a real effect or an artifact of the way the data is prepared. But this is the key to the "magic" feature engineering. For each column, we add a feature counting the number of appearances of the value in the train and test sets. However, it's crucial to exclude the synthetic test data from this count. Again, I haven't really figured out why this works. I tried calculating the counts separately for train and test, and excluding the synthetic test data from the test counts, but although this gave good cross-validation results in training, it didn't seem to work on the test set. So that suggests that the combined train and test count leaks some information from test to train.

### Modelling approach
We model each column separately then combine the predictions. Each column prediction is a blend of 2 lightgbm models and one modified logistic regression.

*Lightgbm 1:*
The model has 2 features, the column value and the frequency count. Maximum depth is 2. We choose the number of iterations separately for each column by cross-validation (since some columns have much more signal that others).

*Lightgbm 2:*
The same as the first model, but instead of the raw frequency count, we used the normalised count, defined as the count divided by the average count of neighbouring points.

*Modified logistic regression:*
We separate the data into 2 sets, frequency 1 and frequency &gt; 1 and fit a separate model for each. In each model, we sort the training data by the value of the column, and divide it into (overlapping) blocks of 20000 samples with a step size of 500. For each block we fit a logistic regression with the single variable c, the column's value, and evaluate it at the midpoint of the column's values. This gives us a prediction of the target for a subset of evenly spaced values of c. Then we linearly interpolate between these. The idea is similar to the Savitzky-Golay filter, but fitting a logistic function instead of a polynomial. Surprisingly, this gives results almost as good as the lightgbm models.

### Combining the predictions from different columns
Chris Deotte's naive Bayes kernel (4) shows that the features are independent within each target class. The approach in the kernel is to multiply all individual column probabilities. However, Julian points out in the comments that this makes the additional assumption that the individual columns are completely independent, not just conditionally independent given the target class. I used the mean of the logits of the original columns, which doesn't require this assumption (more detailed explanation in the pdf on my github page).

### References
1. https://www.kaggle.com/brandenkmurray/randomly-shuffled-data-also-works?scriptVersionId=11467087
2. https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/83882
3. https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split?scriptVersionId=11948999
4. https://www.kaggle.com/cdeotte/modified-naive-bayes-santander-0-899?scriptVersionId=11969430
