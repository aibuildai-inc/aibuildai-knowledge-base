# 9th place solution (a lot of LGBMs blended)

Competition: covid19-global-forecasting-week-5
Rank: #9
Source: https://www.kaggle.com/c/covid19-global-forecasting-week-5/discussion/154951

My approach was quite straightforward: I have trained a lot of very simple LGBM models, blended their predictions, and applied an additional ~linear transformation to optimize the final metric.

[Notebook with my final solution](https://www.kaggle.com/ilialar/w5-solution-2)

**Overview**

- Each model takes as input numbers of new cases and fatalities during the last N days divided by the maximum value of cases/fatalities for this region up to the last day in input and predicts the number of cases or fatalities on Kth day from the last input day (again divided by max value up to date).

- Each model is trained using all available data: all regions and all possible N+K days frames. But different days had different weights – the later the higher. The weight for each particular day is proportional to `(0.9)**(-n)` which means, that last days in the train set had much greater significance during training.

- Each model is just `lightgbm.LGBMRegressor()` with default parameters. I have trained a lot of different models iterating over N (`range(5, 40, 5)`) and K (`range(1,32)`). 

- Then I have used all models to make all possible predictions for each day in the test or validation sets (I have used the last 30 days for validation) and computed min, max, and average prediction for each day.

- It turned out, these predictions were quite pessimistic – predicted values were very high. That’s why I have applied additional linear transformation and deviation factor and computed the final predictions using formulas below (coefficients are optimized on validation). 

```
pred_0.5 = a[0] * mean + a[1] * min + a[2] * max
pred_0.05 = pred_0.5 + (b[0] * mean + b[1] * min + b[2] * max) * dev(lamb_min)
pred_0.95 = pred_0.5 + (c[0] * mean + c[1] * min + c[2] * max) * dev(lamb_max)

```
Where `a`, `b`, and `c` are just linear coefficients. 
`dev(lamb) = np.array([lamb * n for n in range(num_days_to_predict)])` – is used to make predictions for later days less (or more) confident.



What I have not done but probably should (I have joined quite late and just did not have enough time):
- Optimize hyperparameters of LGBM
- Smooth predictions (I see that I have a lot of outliers at some days) by (for example) using 5% and 95% percentiles of predicted values instead of min and max.
- Use other information as inputs or somehow utilize information about connections between different regions.
