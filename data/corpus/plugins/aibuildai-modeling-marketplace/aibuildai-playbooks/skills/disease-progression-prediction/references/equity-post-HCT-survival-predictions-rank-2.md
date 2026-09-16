# 2nd Place Solution

Competition: equity-post-HCT-survival-predictions
Rank: #2
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566522

First of all, I want to thank my teammates for this fun ride.



Since we knew that the data is synthetic and how it was produced, we had a look at the SurvivalGAN paper. As you can see in this image, TimeRegressor is trained together with features and class information. Therefore we split the problem into two:
* efs classification
* efs_time regression

For the regression part, we have trained and ensembled 2 models providing efs as an extra feature. Then for the test and validation sets, we make inference by setting efs=1. With this trick, we estimate how long efs time can be if there is efs. Our models are xgboost and histgbm.


For classification, we use RealMLP, HistGBM and Catboost models with no feature engineering. We ensemble these models by the weights tuned on CV. We then combine it with regression predictions to calculate the risk scores:
R = p(efs=1) * p(efs_time | efs=1)
p(efs_time | efs=1) is sigmoid(-regression_prediction) in our case.

Additionally, we train a neural network which approximates the competition metric and directly predicts the risk scores. It uses regression prediction from tree models and optimizes the approximate competition metric and auxiliary binary classification loss. 

```python
def loss_func(y_risk, y_time, y_efs):
    y_risk, y_time, y_efs = y_risk.ravel(), y_time.ravel(), y_efs.ravel()
    
    loss = []
    for i in range(len(y_risk)):
        if y_efs[i] == 1:
            filt = y_time[i] < y_time
            s = (y_risk - y_risk[i]).tanh()
            loss.append(s[filt])
            
    loss = torch.cat(loss).mean()
    return loss
```

We rank ensemble it with our main pipeline.

**Here is our inference notebook:** https://www.kaggle.com/code/karakasatarik/2nd-place-solution-inference

# Appendix

## AutoGluon
Once the problem is split into two like we did, it was possible to get a gold medal even with automated machine learning.

Here are our experiments using AutoGluon:

| AutoGluon Setting  | OOF Score | Public LB | Private LB |
|--------------------|---------|----------|-----------|
| Medium Quality    | 0.6884  | 0.694    | 0.697     |
| High Quality      | 0.6910   | 0.694    | 0.698     |
| Best Quality      | 0.6921  | 0.695    | 0.699     |

## Postprocess
Anil attempted to improve the efs-classifier model's outputs by tuning a custom sigmoid function on the OOF predictions:

`calibrated_proba = 1 / (1 + np.exp(-beta * (raw_proba - gamma)))`

This calibration led to a ~0.002 improvement in OOF but resulted in a 0.001 decrease on the public LB and a 0.001 improvement on the private LB.
