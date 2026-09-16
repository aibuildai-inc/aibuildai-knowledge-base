# 16th Place Summary

Competition: elo-merchant-category-recommendation
Rank: #16
Source: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82166

## Feature Selection ##

**Just before training**

I used the code from here: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/77537

It is to remove features with different distribution on train and test set.

**After training**

I used the null selection idea (modified).
https://www.kaggle.com/ogrellier/feature-selection-with-null-importances

It is basically a permutation test to adjust the features ranking. Trying to avoid features that work so well on random data.

**After getting the features ranking**

I cut the features with a loop to see what feature size works best. i.e. 90%, 80% .... 10% of features.


## Modified 3-model Model ##

![enter image description here][1]


**Cross-validating the path of the outlier detection -&gt; model without outliers:**

For the original public kernel (training with the manually selected non-outliers (&gt;-33), If the outlier classifier is not good enough, the CV&amp;PB error from the "model without outliers" will be huge.

That's why I came up with a (relatively more stable but with less improvement) modified stacking method to simulate the error "inheriting" from the outlier classifier to the "model without outliers".

I used the predicted non-outliers to train the model without outliers. So that the local cross validation covers the outliers prediction of the test set:

```
all_train_df['outliers'] = all_train_df['prediction'] &gt;= all_train_df['prediction'].quantile(0.9)
```

```
all_test_df['outliers'] = all_test_df['prediction'] &gt;= all_train_df['prediction'].quantile(0.9)
```


## Blending ##
![enter image description here][2]

The final blending consists of 16 models. They are single models, 3-model models, outlier detection models and 2 public kernels.

## Models Summary ##

**Best Single Model**: CV: 3.63964, Public: 3.690, Private: 3.606

**Best 3-model Model**: CV: 3.63469, Public: 3.685, Private: 3.604


  [1]: https://i.imgur.com/6otyev8.png
  [2]: https://i.imgur.com/xTZjWms.png
