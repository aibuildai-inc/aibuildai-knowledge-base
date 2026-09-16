# 4th Place Solution for the Child Mind Institute — Problematic Internet Use competition

Competition: child-mind-institute-problematic-internet-use
Rank: #4
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552861

I can’t believe I secured 4th place in this competition. This is my first time participating in a Kaggle competition, and I’m so lucky that I managed to place so well. I’m grateful for all the support from the community and the amazing resources available on Kaggle. I’m excited to continue learning and taking on new challenges in the future!

## Overview

We focused heavily on data preprocessing. First we drop all rows with ambigious sii. Then after reviewing several notebooks and conducting our own research, we selected key features to handle missing values. For less important columns, we chose not to fill missing values as we found that doing so worsened the results, likely due to lack and unreliable data. For important columns, missing values were filled using a submodel with inputs from other reliable columns (such as demographic data or pre-filled columns). This sub-model could be linear regression, logistic regression, or KNN, depending on the case. We also added weights to certain columns like CGAS-CGAS_Score and SDS-SDS_Total_Raw (which will be explained later)

We proceeded to feature engineering, where we combined features that we believed had clear relationships, such as age and BMI, or SDS and CGAS…

For our final model, we employed a stacking approach combining three high performing models: CatBoost, LightGBM, and XGBoost. We train our models in 5 folds of data, and then optimized sii decision rounding threshold

## Details

### Most important features
List of features we thought were the most important:
- Age
- Physical columns (BMI, weight, height, waist circumstances)
- Internet use hours
- SDS (raw) 
- CGAS score

### Handle missing values
Here are how we fill the missing values for these columns:
Age, Sex, Demos-Season ----- knn -----> Physical Weight, Height
Weight, Height ----------> BMI
BMI, Weight ----- Linear regression -----> Waist circumstances
Age, SDS-Season ----- knn -----> SDS-Total-Raw
Age, Sex, SDS-Total-Raw, Internet-Season ----- Logistic Regression -----> Internet hours use

💡Note: We didn't fill CGAS score because we can't find a strong enough relationship between it and any other columns beside Age, but it's still an important feature

For other columns, include the time series columns, we will remove the outliers and incorrect data, or just completely removed columns that were deemed unhelpful

### Add some weights for CGAS score column and SDS score raw column
After observing:
- There are no participants with an SII score of 3 who have a CGAS score > 80.
- There are no participants with an SII score of 3 who have an SDS score < 35.

This might indicate that a CGAS score of 80 and an SDS score of 35 could serve as effective thresholds for predicting who has severe problematic internet use (PIU) and who does not.

Therefore, we decided to assign weights to these two columns. The weights are calculated using a sigmoid function. The characteristic of the sigmoid function is that the closer the values are to the threshold, the steeper the curve becomes, allowing for clearer differentiation.

For example, the weight for the CGAS score is calculated as follows:
$$
\text{CGAS_Weight}(cgas, a, b) = \frac{1}{1 + e^{-a \cdot (cgas - b)}}
$$

The plot:


When the CGAS score is closer to 80, the curve becomes steeper. This means the weight for the CGAS score changes more, enhancing its ability to aid in classification.

We then multiply the CGAS score by its weight to create a new feature, Weighted_CGAS_Score, which will be used in the feature engineering process.

The same technique is applied to the SDS score, resulting in Weighted_SDS_Score.

### Feature Engineering
We did not employ any particularly advanced techniques here, we just combine columns that we feel were related to each other. Important features are prioritized more.

### Modeling and training
Train and validation score results for individuals and stacking model, calculated by quadratic cohen kappa score:
|| XGB | LGB | CatBoost | Ensemble |
| --- | --- | --- | --- | --- |
| Train | 0.6429 | 0.8066 | 0.5472 | 0.6138 |
| Validation | 0.3846 | 0.3909 | 0.3857 | 0.3912 |

### What were tried but didn't work
- We added Tabnet for ensemble model but it never did great
- We tried to predict the PCIAT-PCIAT_Total column at first and then map it to SII but it also yielded poor results. Our performance improved when we switched to predicting SII directly and applied a threshold tuning technique to optimize the rounding threshold.

## Sources
Best EDA ever https://www.kaggle.com/code/antoninadolgorukova/cmi-piu-features-eda
Time series data EDA and threshold tuning methods from https://www.kaggle.com/code/ambrosm/piu-eda-which-makes-sense

Thank you for reading till the end
