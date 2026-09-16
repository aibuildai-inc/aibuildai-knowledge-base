# 2nd Place Solution - 7 models, but 1 was also enough

Competition: playground-series-s5e11
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s5e11/writeups/2nd-place-solution-7-models-but-1-was-also-enou

My best solution is a ridge ensemble of 7 models, 5x LGBM, 1x TabM, 1x RealMLP. My other submission, which was my best single LGBM model, also achieves 2nd place. Simplified version of that model can be found in [this](https://www.kaggle.com/code/angelosmar1/s5e8-lgbm-cv-0-92813-2nd-place) notebook.

## Features

Apart from the initial columns, I add all initial columns with cardinality > 7:
* Target encoded
* Count encoded
* Target encoded with targets from the original dataset

The key for a high CV was to discretize/bin the high cardinality columns `annual_income` and `loan_amount` in different ways before target encoding them. Each of the following ways contributed to a better CV:

* quantile binning
* uniform binning
* using `round` and the `//` (integer division) operator
* using `.astype(int)` to discard the decimal part

Furthermore, i include digit features from all numeric columns and combinations of digits from some numeric columns.
Finally, i add ratios of counts between the training data and original data (how much each value is oversampled or undersampled compared to original).

Among things that did not improve CV were interaction features and adding original data as rows.

## Parameters

LGBM model improved a lot with heavy regularization parameters (low `max_depth`, `colsample` etc).

```
learning_rate=0.01
max_depth=4,
subsample=0.5
colsample_bytree=0.2
lambda_l2=15.0
lambda_l1=10.0
```

## Best Ensemble

| Model  | 5-fold CV | Notes |
|------|------|------|
| LGBM    | 0.92813    | Best single model    |
| LGBM     | 0.9281    | Original data added as extra rows once every fold    |
| LGBM     | 0.9278    | Original data added as extra rows once every fold, less features   |
| LGBM    | 0.9277    | Logits initialised from TabPFN trained on original data, idea from [here](https://www.kaggle.com/competitions/playground-series-s5e11/discussion/614986)    |
| LGBM    | 0.9276    | Logits initialised from Autogluon trained on original data with `preset='extreme'` option, idea from [here](https://www.kaggle.com/competitions/playground-series-s5e11/discussion/614986)   |
| TabM    | 0.9277    | -    |
| RealMLP    | 0.9271    | -    |


## Conclusion

Congratulations to @mahoganybuttstrings for winning. Thanks to all Kagglers who contribute in discussions and notebooks. Exploring Playground competitions is fun and educational.
