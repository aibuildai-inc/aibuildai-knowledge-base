# 19th Place Solution for the "ICR - Identifying Age-Related Conditions" Competition

Competition: icr-identify-age-related-conditions
Rank: #19
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/431099

Hey Kagglers,
I guess mine is one of the simplest approach for this competition.

<br />

**Solution:** 
https://github.com/SunilGolden/Kaggle-ICR

<br />

**Approach**
- Under sampled training data
- Imputed null values with zero
- Encode categorical column using Ordinal Encoder
- Scaled other columns using Min Max Scaler
- Used k-fold cross validation to evaluate TabPFN, XGBoost, CatBoost, HGBoost, Light GBM, Random Forest, AdaBoost, GBM, SVM models and a few versions of their emsembles with balanced log loss
- Finally, I trained XGBoost, CatBoost, HGBoost, Light GBM, Random Forest, and GBM models and then ensembled them.

<br />
