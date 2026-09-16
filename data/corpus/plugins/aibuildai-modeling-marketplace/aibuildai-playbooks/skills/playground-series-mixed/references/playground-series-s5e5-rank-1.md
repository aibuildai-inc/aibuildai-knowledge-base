# 1st Place - GPU Hill Climbing!

Competition: playground-series-s5e5
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582611

Thanks Kaggle for another fun playground competition. This month was different than past months because the features were all numeric without NANs.

# GPU Hill Climbing
My final solution is the result of feeding hundreds of GBDT, NN, and NVIDIA cuDF cuML models into my GPU Hill Climbing starter notebook [here][1]. Over the month of May 2025, I used the speed of GPU and NVIDIA cuDF cuML to build as many diverse models as possible.

Hill climbing is great because it automatically selects models for us. From the hundreds of candidate models, hill climbing selected the following 7 models:

| Weight | Model | Notes | CV score |
| --- | --- | --- | --- |
| 1/12 | XGBoost | cuML target encoded features1 | 0.06084 | 
| 1/12 | XGBoost | cuML target encoded features2 | 0.06061 |
| 1/12 | XGBoost | cuML target encoded features3 | 0.06053 |
| 1/4 | XGBoost | product features | 0.05951 |
| 1/6 | CatBoost | binned features and groupby features | 0.05937 |
| 1/6 | NN over LinearRegression | NN is [here][2] | 0.05999 |
| 1/6 | XGB over NN | NN is [here][2] | 0.05989 |

# Final Ensemble CV and LB Scores
The final hill climbing ensemble has 
* RMSLE **CV = 0.05880**
* **Public LB = 0.05677**
* **Private LB = 0.05841**

# XGBoost with cuML Target Encoder (CV=0.06XX)
In my final ensemble 25% of the weight is XGBoost with NVIDIA cuML TargetEncoder features. This demonstrates that diversity is more important than single model CV score. My XGBoost with TE features each have a poor CV score of 0.06XX but they improved the final ensemble `CV 0.05890 => 0.05880` and `Private LB 0.05847 => 0.05841`

# XGBoost with Product Features (CV = 0.05951)
For each feature, i created a `log1p` version, i.e. `df[f'log1p_{c}'] = log1p( df[c] )`. Then i created all products, divisions, sums, and differences between all pairs of features.

# CatBoost with Binned Features and GroupBy Features (CV=0.05937)
CatBoost loves categorical features, so I converted each numerical feature into 9 equal width binned values. I also created `log1p` versions of all features and converted those into 9 binned values. Afterward I created combinations of all pairs of columns. The resultant new columns had 81 unique values and were also categorical. We then use `cat_features = CATS`.

For groupby features, I would pick a group of people like `Sex and Age` using the bins, then I would compute each person's `z-score` about their `height`. And then their `z-score` about their `weight`, i.e. for a male in their 40's how does their weight compare with other males in their 40's. 

I created 26 of these features. I would make groups from 1 to 3 features then compute `z-score` of another feature. Another example is `[ ["Sex","Weight_bin","Body_Temp"], ["Heart_Rate"] ],`. This means i make groups from Sex, Weight, Body_Temp and compute `z-score` for Heart_Rate.

# NN over Linear Regression (CV=0.05999)
I trained my public notebook NN on the residuals from a NVIDIA cuML LinearRegression model. This improved CV `0.0608 => 0.0599`. The linear regression model does a great job capturing the linear relationships in the data. And then it helps the NN learn these relationships better.

* train LinearRegression with 5 Kfold seed 42. Make OOF predictions and test PRED predictions.
* make a **new target** with `new_target = old_target - LinearRegression_OOF`
* train NN with train data and **new target** (with 5 Kfold seed 42)
* infer NN on test, then `final_pred = NN_PRED + LinearRegression_PRED`.

# XGB over NN (CV=0.05989)
I trained my XGB with Product Features on the residuals from my public NN notebook. This did not improve the XGB CV score but it made a new diverse model that improved my hill climbing ensemble CV score. We implement this using similar bullet points as above in "NN over Linear Regression" section.

# Icing on the Cake - Retrain Using 100% Train
I use the following trick in all my Kaggle competitions. My OOF are 5-Kfold and we use hill climbing to find the model weights from these OOF. Then we retrain all the models using 100% train data and use fixed number of iterations equal to 25% ( which equals to `1/(K-1)` ) more than average number of iterations from 5-Kfold early stopping. Then we take a weighted average of these 100% test preds using the weights found from hill climbing based on 5-Kfold OOF. This gives a nice boost in every Kaggle competition! (I also train K of these using a different seed each time and average the predictions).

[1]: https://www.kaggle.com/code/cdeotte/gpu-hill-climbing-cv-0-05930
[2]: https://www.kaggle.com/code/cdeotte/nn-mlp-starter-cv-0-0608
