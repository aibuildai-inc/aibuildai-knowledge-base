# [16th place solution] Features Diversity and Ensemble

Competition: amex-default-prediction
Rank: #16
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/347858

We would like to thank the organizers and the Kaggle community for providing such a great competition. 
I would like to thank @shivamcyborg and @eventhorizon28 for their support and contribution, our team's collective hard work helped us achieve this position. 
Special thanks to @raddar, @roberthatch, @cdeotte, @jiweiliu, @ragnar123  for the awesome work they published without their analytics, This competition would have a different direction from what  it is now.

Here brief Explanation of our solution.

*FEATURE ENGINEERING*
*DIVERSITY IN MODELS*

#### Feature Engineering
We used different features for various model training (mean, std, and last features were common). We trained our models in three ways.
1. using only HMA(hull moving average) features
2. using only diff(features)
3. using HMA + diff features ( worked only with cat boost)

Using all the diff features was not the right call for some of the models like NN and XG Boost as they were introducing some leakage while training and CV and LB  didn't correlate at all. For us, HMA features proved to be much better featured than diff features.

#### Models
We used various models that include, LGBM, XG Boost, Cat Boost, and Neural Networks with 2 different architectures and TABNET.

Here are our best single model scores
| models | Cross Val | private LB | public LB | Description | Core Features | 
| --- | --- | --- | --- |
| LGBM |.7973 |0.80687 |0.79906|3 models with different seed+ 2 public models |Diff+Last|
| XG Boost |.7972 |0.80639|0.79718|3 models with different seed+ 1 public model |HMA+Last|
| CAT Boost |.7952 |0.80468|0.79614|3 models with different seed |HMA+diff+Last|
| NN-1 |.7923 |0.80190|0.79240|3 models with different seed |HMA+Last|
| NN-2 |.7921 |0.80186|0.79188|2 models with different seed |diff + Last|
|TABNET |.7933 |-|-|single model trained last day to introduce diversity |HMA + Last|

#### ENSEMBLE
Since we had such a great CV LB correlation, we used Optuna to choose the ensemble weights and performed rank ensemble for our submissions. However, since some good public models didn't have oofs predictions we had to give weights to them manually.

Things that we were not able to try:
1. Using B_29  predicting the missing values and then using them as features
2. LGBM with HMA feature that would be our best public model
3. pseudo-labelling using just private LB data


Notebooks that helped us:

- https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793
- https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977
- https://www.kaggle.com/code/roberthatch/xgboost-pyramid-test-predictions
- https://www.kaggle.com/code/ambrosm/amex-keras-quickstart-1-training
- https://www.kaggle.com/code/werus23/amex-keras-with-tpu
- https://www.kaggle.com/code/raddar/understanding-na-values-in-amex-competition
- https://www.kaggle.com/code/raddar/the-data-has-random-uniform-noise-added
- https://www.kaggle.com/code/jiweiliu/amex-catboost-rounding-trick.
- https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793
