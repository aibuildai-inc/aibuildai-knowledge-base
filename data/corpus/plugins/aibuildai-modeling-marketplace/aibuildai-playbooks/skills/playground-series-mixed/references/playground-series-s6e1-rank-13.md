# 13th place - DiversitySlop

Competition: playground-series-s6e1
Rank: #13
Source: https://www.kaggle.com/c/playground-series-s6e1/writeups/13th-place-diversityslop

Congratulation to the winners and thanks for discussions/code: @include4eto , @omidbaghchehsaraei  , @cdeotte , @tilii7 

I threw 173 models at this along with hillclimbing and applied Ridge regression on top of it. The ensemble worked better than any single model, but I'm still not sure I squeezed out every last decimal. The hardest part was score range bias, models kept overpredicting low scores and underpredicting high ones. I tried weighted objectives, per-group models, and various post-processing tricks. Some helped. Some didn't. But one thing was clear to me, that i had to find every other way to create a diverse model for hillclimbing/stacking to work .

---

## What worked

### Asymmetric weighted loss

The standard MSE loss treats all errors equally. But my error analysis showed the model was terrible at extremes: overpredicting low scores by 10+ points, underpredicting high scores by 8+ points. I needed the model to pay more attention to these ranges.

The solution: a custom XGBoost objective that scales the gradient by target-dependent weights.

```python
def weighted_squared_error(predt: np.ndarray, dtrain: xgb.DMatrix):
    """
    Weighted Squared Error objective with edge-boosted weights.
    """
    y = dtrain.get_label()

    # Define edge weights based on true target values
    edge_weight = np.ones_like(y, dtype=np.float64) * 2.0
    edge_weight[y <= 27] = 8.0                    # 4x boost for very low scores
    edge_weight[(y > 27) & (y <= 35)] = 4.0       # 2x boost for low scores
    edge_weight[(y >= 84) & (y < 92)] = 3.0       # 1.5x boost for high scores
    edge_weight[y >= 92] = 6.0                    # 3x boost for very high scores

    # Gradient: d/d(pred) of 0.5 * w * (pred - y)^2 = w * (pred - y)
    grad = (predt - y) * edge_weight

    # Hessian: d^2/d(pred)^2 of 0.5 * w * (pred - y)^2 = w
    hess = edge_weight.copy()

    return grad, hess
```

 I set them based on error analysis: score range 19-27 had mean error of +10.2 (severe overprediction), so it gets 4x the attention. Score range 92-100 had mean error of -8.4 (underprediction), so 3x attention there.

### Disagreement-based model selection

A model with mediocre RMSE might be exactly what the ensemble needs if it makes different mistakes than the other models.

I used a greedy selection algorithm that maximizes prediction disagreement:

1. Start with the best single model (safe baseline)
2. Add the model that disagrees most with the current selection
3. Repeat until diminishing returns (detected via elbow method)

The disagreement between two models is just the mean absolute difference in their predictions. Simple, but it captures models that predict differently in different situations. All models ( 4 or 5 ) selected via this methods were used for stacking, which itself was fed into a hillclimbing method  

Since meta-learner learns when to trust which model based on input features , instead of just tuning hyperparameters for lowest RMSE, I started deliberately creating diverse models:

- **Huber loss models**: Robust to outliers, makes different errors than MSE-trained models
- **Quantile regression models**: Trained at different quantiles (0.25, 0.5, 0.75), each optimizes for different parts of the distribution
- **Asymmetric error models**: Different weight schemes for different population segments

A model trained with 8x weight on low scores makes systematically different predictions than one trained with uniform weights. That diversity is exactly what the meta-learner needed.

### Outlier classifier stacking

I trained a binary classifier to predict whether a sample is an outlier (target more than 1.5 std from mean). The classifier's probability output became an additional feature for the final model.

 If the classifier thinks a sample is likely to be an outlier, the regression model should treat if as a special case and probably find different patterns for it

Stacking the outlier probability with the best regression models improved the ensemble. 

### Per-group models

I split the data by `internet_access` and trained separate models for each group. The error patterns differed between groups, so specialized models made sense. The meta-learner then learned to route predictions appropriately.
