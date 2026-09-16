# 8th Place Solution

Competition: child-mind-institute-problematic-internet-use
Rank: #8
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552760

First, I'd like to thank all the hosts and Kaggle staff for organizing such an interesting and practical competition!
I greatly enjoyed the process of trial and error for handling noisy and missing data!

# OverView

- I concentrated on improving CV, not LB due to the small number of data and unclear distribution of missing values.
- I extracted features by Null Importance to improve robustness.
- I imputed numerical missing features by Iterative Imputer (estimator=Bayesian Ridge).
- Ensemble in 3 different models. 
 - train.csv + descriptive actigraph features
 - tain.csv(+ Feature Engineering) + descriptive actigraph features
(+ time_of_day divided into 4 groups [0-21600, 21600-43200, 43200-64800, 64800-86400])
 - train.csv(only)
- Regression models
 - objective: mse
 - models: lightgbm, xgboost, catboost
- I calculated QWK with 3 Seeds×5 CV(StratifiedKFold) to reduce fluctuations.
- QWK thresholds optimized by 3 Seeds×5 Stratified CV to improve robustness.

# Feature Extraction

I used Null Importance based on lightgbm gain importances to reduce features as much as possible while considering the trade-off CV.
The number of features after reduction is shown below.

| Model | Feature Nums |
| --- | --- |
| train.csv + descriptive actigraph features | 13 |
| tain.csv(+ Feature Engineering\*) + descriptive actigraph features <br> (+ time_of_day divided into 4 groups [0-21600, 21600-43200, 43200-64800, 64800-86400] \*\*) | 40 |
| train.csv(only) | 14 |

\* Feature Engineering is the same as many public notebooks. I also calculated the absolute values of the actigraphy data and `df['XYZ']=np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)`.
\*\*  I calculated descriptive actigraph features for each divided group. 

## Extracted Features

**train.csv + descriptive actigraph features**
` ['Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight', 'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_PU', 'BIA-BIA_Activity_Level_num', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday', 'light_max']`

**tain.csv(+ Feature Engineering)+ descriptive actigraph features <br> (+ time_of_day divided into 4 groups [0-21600, 21600-43200, 43200-64800, 64800-86400])**
`['Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference', 'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_PU', 'PAQ_A-PAQ_A_Total', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday', 'X_25%_0-21600', 'Y_std', 'Z_mean_0-21600', 'Z_min_43200-64800', 'Z_75%_64800-86400', 'enmo_50%_43200-64800', 'non-wear_flag_mean_0-21600', 'light_std', 'light_max_21600-43200', 'XYZ_50%_21600-43200', 'XYZ_mean_43200-64800', 'XYZ_mean_64800-86400', 'XYZ_std_64800-86400', 'XYZ_50%_64800-86400', 'abs_X_50%_21600-43200', 'abs_X_25%_64800-86400', 'abs_X_75%_64800-86400', 'abs_Y_75%_0-21600', 'abs_Y_min_21600-43200', 'abs_Y_75%_64800-86400', 'abs_Z_75%_21600-43200', 'abs_Z_75%_43200-64800', 'abs_anglez_min_43200-64800', 'BMI_Age', 'Internet_Hours_Age', 'Muscle_to_Fat']`

**train.csv(only)**
`['Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight', 'Fitness_Endurance-Max_Stage', 'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_PU', 'BIA-BIA_Activity_Level_num', 'PAQ_A-PAQ_A_Total', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday']
`

# Imputation of missing values

I adopted IterativeImputer and the default Bayesian Ridge as an estimator to impute missing values. I also tried GBDT and RandomForest, but the CV was lower than Bayesian Ridge.
CV was improved the most when training the imputer on data with 30 or fewer missing values. Also, training on data with missing 'sii' would have reduced CV, so we did not use those data.

# Models

I adopted a simple weighted blend, and weights were determined to maximize CV.

| Model | weights |
| --- | --- |
| train.csv + descriptive actigraph features |  lgb: 0.2, xgb: 0.2 |
| tain.csv(+ Feature Engineering) + descriptive actigraph features <br> (+ time_of_day divided into 4 groups [0-21600, 21600-43200, 43200-64800, 64800-86400]) | xgb: 0.2 |
| train.csv(only) | lgb: 0.2, cat: 0.1, xgb: 0.1 |
  
<br/>
  
| Model | CV |
| --- | --- |
| Baseline | 0.475 |
| Submission Model | 0.501 |

The baseline model is a single model of lightgbm without imputation of missing values having all train.csv + descriptive actigraph features.(Hyper parameters of models were optimized)

# Optimization of Thresholds

To ensure robust thresholds, they were calculated by the Nelder-Mead method from the entire 3 Seeds×5CV.

# What didn't work

* I trained the imputer on data with missing sii, but CV decreased.
* Huber and MAE were used as objective functions but did not improve CV.
* CNN and LSTM were applied to actigraphy data, but learning did not proceed well.

# What I didn't try

* pseudo-labeling
* custom-objective

　<br/>

Thank you for reading!

---

You can check the solution code from the below links.

training: https://github.com/beagledeveloper/Child_Mind_Institute-Problematic_Internet_Use_8th_Place

inference: https://www.kaggle.com/code/kzkknmt/8th-solution-inference-note
