# 10th Place Solution

Competition: predict-student-performance-from-game-play
Rank: #10
Source: https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420132

I respect all of you for your tough and long term fight and I am glad that we were able to fight together. I also want to thank my teammates ( @tereka, @deepkun1995, @ryotak12, @yurimaeda) for their hard work.

I'm happy because this is the first time I got a gold medal.

# Overview
 We're not doing anything special in our solution. We used 1 NN, 1 LightGBM and 4 XGBoost with various features for the Stage 1, and MLP and Logistic Regression stacking for the Stage 2. We used average and threshold optimization for the Stage 3.
 - CV: 0.70573
 - Public LB: 0.706
 - Private LB: 0.702

[psp_solution_overview].png?generation=1688311769280377&alt=media)

## Code
- Inference: https://www.kaggle.com/code/shu421/psp-10thsolution-public0706-private0702/notebook
- shu421 XGBoost and Stacking Training: https://github.com/shu421/Kaggle_PSP_10thSolution

# Models
## Stage 1: XGBoost (shu421 part)
I created XGBoost for each level_group. The base features are not so different from those in the public code. It is an aggregate feature of elapsed_time_diff and hover_duration, and other numerical features. However, in addition to these, I used previous level_group features and predicted probability as current level_group features.  
I used numpy and numba to create them. Initially, I had used polars, but I switched to numba which is my teammate @yurimaeda 's approach. The submission time was significantly reduced from 2 hours with polars to just 13 minutes with numpy and numba. I used 5-StratifiedGroupKFold as cross-validation strategy.
- CV: 0.70111
- Public LB: 0.702
- Private LB: 0.699

[psp_solution_xgboost].png?generation=1688311791487937&alt=media)


## Stage 2: Stacking
We created MLP and Logistic Regression for each question. Thus, there are 18 models each, and the output dimension of each model is (n_samples, 1).
Since stacking was very easy to overfit, we kept the model architecture simple.
Here is the code for MLP.

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.head(x)
        return x
```


## Stage 3: Threshold Optimization
We take the average of the predictions of the 2 models in the Stage 2 and optimize the threshold for each question.

```python
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import minimize


def f1_score_macro_for_thresholds(y_true, y_pred_prob, thresholds):
    y_pred_binary = (y_pred_prob > thresholds).astype(int)
    score = f1_score(y_true.flatten(), y_pred_binary.flatten(), average="macro")
    return score


def optimize_thresholds(y_true, y_pred_prob, method="Powell"):
    n_labels = y_pred_prob.shape[1]
    init_thresholds = np.full(n_labels, 0.6)

    objective = lambda thresholds: -f1_score_macro_for_thresholds(
        y_true, y_pred_prob, thresholds
    )
    result = minimize(
        objective, init_thresholds, bounds=[(0, 1)] * n_labels, method=method
    )

    return result.x
```

We tried some optimization methods, but Powell worked best.
This method improved CV by 0.008.

# What worked
- feature engineering
  - elapsed_time_diff and hover_duration agg features was important
- threshold optimization
- ensemble
- lstm + transformer(ryota part)
- [sort_frame](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/416963)

# What didn't work
- 1D CNN
- stacking(below methods seemed to be overfitting)
  - CNN(1D/2D)
  - RNN
  - level_group preds
- datetime agg features
- use NN embedding as gbdt features
- TimeSeriesClustering (elapsed_time_diff)
- additional data
