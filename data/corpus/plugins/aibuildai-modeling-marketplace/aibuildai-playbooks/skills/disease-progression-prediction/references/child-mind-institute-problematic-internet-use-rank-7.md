# Private 7th place solution

Competition: child-mind-institute-problematic-internet-use
Rank: #7
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552625

Hello, everyone. 

First, I want to thank the organizers for hosting this competition. Working with real-world data was challenging but provided an excellent opportunity to enhance my technical skills for practical applications.

Below, I share my solution:

### Model

- I used this baseline: [CMI Single LGBM CV 0.471 LB 0.460](https://www.kaggle.com/code/greysky/cmi-single-lgbm-cv-0-471-lb-0-460).
- The primary model is LGBM(, with XGBoost included in the ensemble).
- Missing values were handled using median imputation, calculated only on the training data for each fold, and applied to the validation/test sets.
- Sequential data statistics were processed as in the public notebook but optimized for speed using Polars.
- I used Tweedie loss as the primary loss function(, while classification was also included in the ensemble). → **important**
- Predictions for data where 'sii' was not present were generated and used as pseudo-labels for training. → **important**
- Features defined as categorical integers in the data dictionary were used both categorically and numerically.

### Threshold Optimization → **important**

Threshold optimization was performed on each fold's training data and applied to the validation/test sets. Using percentiles further improved both accuracy and robustness. Initial optimization values were derived from the discussion [here](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/discussion/551533).

```python
def calc_initial_th(y, y_pred):
    tmp_df = pd.DataFrame({"sii": y, "prediction": y_pred})
    oof_initial_thresholds = (
        tmp_df.groupby("sii")["prediction"].mean().iloc[1:].values.tolist()
    )

    oof_threshold_percentiles = [
        (tmp_df["prediction"] <= threshold).mean()
        for threshold in oof_initial_thresholds
    ]

    return oof_threshold_percentiles

oof_initial_thresholds = calc_initial_th(y, y_pred)

self.optimizer = minimize(
    eval_preds_percentile,
    x0=oof_initial_thresholds,
    args=(y, y_pred),
    method="Nelder-Mead",
    bounds=[(0, 1), (0, 1), (0, 1)],
)
```

### CV Strategy and Model Evaluation

I used 5-fold StratifiedKFold. However, as you may know, scores can vary significantly depending on the seed. In the aforementioned notebook, the results were averaged across 10 seeds for submission. Similarly, in validation, I evaluated the voting results from 10 seeds as the score for each experiment. I also evaluated the 10-seed results in validation and optimized parameters for each seed using Optuna.

For example, one submission produced the following seed-dependent score variations:
```
[0.4924600384356978,
 0.480933892862644,
 0.4896986007283455,
 0.4848801590924426,
 0.47931930137389445,
 0.48348085393094636,
 0.48327061771577756,
 0.47943771383186834,
 0.47465614619853563,
 0.4909707394562764]
```
I monitored both the voting score and the average of the individual scores. For instance, in the example above, the voting score is 0.49218340134866767, and the average score is 0.4839108063626429. For submissions, I performed a voting ensemble using 5 folds × 10 seeds per model.


While I aimed to make the evaluation and learning process as robust as possible, I think my ranking still depended heavily on luck.
Thank you for reading!
