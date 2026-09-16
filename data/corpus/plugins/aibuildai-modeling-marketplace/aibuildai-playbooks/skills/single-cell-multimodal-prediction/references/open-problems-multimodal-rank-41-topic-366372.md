# Private 41st Solution summary

Competition: open-problems-multimodal
Rank: #41
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/366372

Thank you for all kagglers and organizers for this competition.

This note is a record of my work on the competition.

This was a competition where the Private test data was from a future date that did not exist in the training data, and the so-called domain generalization performance was being tested.
On the other hand, there was an element of variation by date, and it was expected that it would be undesirable to completely ignore the date feature.

First, we conducted adversarial training (a task to classify training and test data) and found that Citeseq was capable of 99% classification, and we were concerned that training with this feature set would result in overtraining on the training data.
However, when we reduced the number of features to reduce the accuracy of Adversarial training, the score of Public LB also dropped significantly.

Therefore, we decided to devise some kind of biological features and to improve generalization performance through model variation.


## ✨ Result

- Private: 0.769
- Public: 0.813



## 🖼️ Solution


### 🌱 Preprocess

- Citeseq
    - The input data was reduced to 100 dimensions by PCA.
    - On the other hand, the data of important columns were preserved.
    - [Ivis unsupervised learning](https://bering-ivis.readthedocs.io/en/latest/unsupervised.html) was used to generate 100 dimensional features.
    - In addition, we added the sum of mitochondrial RNA cells to the features.
    - Cell type in Metadata was added to the features.

- Multiome
    - For each group with the same column name prefix, PCA reduced the number of dimensions to approximately 100 each.
    - Ivis unsupervised learning was used to generate 100 dimensional features.

### 🤸 Pre Training

- Adversarial training (a task to classify training data and test data) is performed and the misjudged training data is used as good validation data.
- Prediction of Cell type for Multiome is performed and added to the features.


### 🏃 Training

- StratifiedKFold with good validation data as positive labels.
- Pearson correlation coefficient was used for the Loss function. XGBoost was implemented as described below.
- TabNet also performed pre-training. (In this competition, pre-training was more accurate.)

### 🎨 Base Models

- Citeseq
    - TabNet
    - Simple MLP
    - ResNet
    - 1D CNN
    - XGBoost
- Multiome
    - 1D CNN


Citeseq scored well with an ensemble of various models.
On the other hand, Multiome had a strong 1D CNN and did not score well with ensembles of other models, so only the 1D CNN was used.

### 🚀 Postprocess

- Since the evaluation metric is the Pearson correlation coefficient, each inference result (including OOF results) was normalized before ensemble.
- Optuna was used to optimize the ensemble weights. Good validation data was used as the evaluation metric.
- Ensemble with Public Notebook x2 and teammate submissions.


## 💡 Tips


### Pearson Loss for XGBoost

XGBoost does not provide a Pearson Loss Function, so I implemented it as follows.
However, this implementation is slow in learning, and I have the impression that I would like to improve it a little more.

```
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb


def pearson_cc_loss(inputs, targets):
    try:
        assert inputs.shape == targets.shape
    except AssertionError:
        inputs = inputs.view(targets.shape)

    pcc = F.cosine_similarity(inputs, targets)
    return 1.0 - pcc


# https://towardsdatascience.com/jax-vs-pytorch-automatic-differentiation-for-xgboost-10222e1404ec
def torch_autodiff_grad_hess(
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], y_true: np.ndarray, y_pred: np.ndarray
):
    """
    Perform automatic differentiation to get the
    Gradient and the Hessian of `loss_function`.
    """
    y_true = torch.tensor(y_true, dtype=torch.float, requires_grad=False)
    y_pred = torch.tensor(y_pred, dtype=torch.float, requires_grad=True)
    loss_function_sum = lambda y_pred: loss_function(y_true, y_pred).sum()

    loss_function_sum(y_pred).backward()
    grad = y_pred.grad.reshape(-1)

    # hess_matrix = torch.autograd.functional.hessian(loss_function_sum, y_pred, vectorize=True)
    # hess = torch.diagonal(hess_matrix)
    hess = np.ones(grad.shape)

    return grad, hess


custom_objective = partial(torch_autodiff_grad_hess, pearson_cc_loss)


xgb_params = dict(
    n_estimators=10000,
    early_stopping_rounds=20,
    # learning_rate=0.05,
    objective=custom_objective,  # "binary:logistic", "reg:squarederror",
    eval_metric=pearson_cc_xgb_score,  # "logloss", "rmse",
    random_state=440,
    tree_method="gpu_hist",
)  # type: dict[str, Any]

clf = xgb.XGBRegressor(**xgb_params)
```


## 🏷️ Links

- [My Solution](https://github.com/IMOKURI/kaggle-multimodal-single-cell-integration)
