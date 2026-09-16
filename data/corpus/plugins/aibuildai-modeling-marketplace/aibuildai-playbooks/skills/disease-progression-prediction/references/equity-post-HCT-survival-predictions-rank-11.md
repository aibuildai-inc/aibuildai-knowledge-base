# 11th Place Solution

Competition: equity-post-HCT-survival-predictions
Rank: #11
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566624

First of all, my teammate @siddhant1104 and I would like to thank the organizers for hosting this competition and congratulate all the winners. It was such a hardship to decide focus on improving whether the LB score or the CV score because when the CV score was increased, our LB scores mostly got worse. Our solution is weighted ensemble of two separate approaches.

## Sercan's Pipeline

- Notebook: https://www.kaggle.com/code/sercanyesiloz/cibmtr-ensemble-learning?scriptVersionId=226093818
- Ensemble of NN with Pairwise Ranking Loss and GDBT pipelines
- Sample weights by **race groups**
- Ordinal encoding for <code>dri_score</code> and <code>conditioning_intensity</code> (tuned with optuna)
- Crossing <code>donor_age</code> and <code>age_at_hct</code> features to extract some
- Cyclic year features (<code>sin_year</code> and<code>cos_year</code>)
- Kaplan Meier, Nelson Aalen and Cox target transformations
- 4 LightGBM and 6 CatBoost models
- GBDT Cross-Validation Setup -> Stratified 10 Folds by **race groups**
- Neural Networks Cross-Validation Setup -> Stratified 5 Folds by **race groups** and **age_at_hct==0.44**
- Predictions were scaled before blending

### Ordinal Encoding
```
dri_score_mapping = {
    "High": 0.6850752146154907,
    "High - TED AML case <missing cytogenetics": 0.18589473149703015,
    "Intermediate": 0.5683465067841215,
    "Intermediate - TED AML case <missing cytogenetics": 0.7708720693082163,
    "Low": 0.9586424711654987,
    "Missing disease status": 0.6831791561653417,
    "N/A - disease not classifiable": 0.7166435651957048,
    "N/A - non-malignant indication": 0.8821201547093761,
    "N/A - pediatric": 0.49866306284678735,
    "TBD cytogenetics": 0.9411056819278409,
    "Unknown": 0.1890854786067684,
    "Very high": 0.5377767827330516
}

conditioning_intensity_mapping = {
    "Unknown": 0.6026915942898587,
    "MAC": 0.02153075067313332,
    "RIC": 0.8995437792670338,
    "NMA": 0.9211477712757186,
    "TBD": 0.7388173559148422,
    "No drugs reported": 0.22412092165882558,
    "N/A, F(pre-TED) not submitted": 0.32745500022610163
}

```


## Siddhant's Pipeline

- Ensemble of NN with Pairwise Ranking Loss and ML (GBDT and Linear Models) pipelines
- Sample weights by **race groups**
- Ordinal encoding for many features
- Frequency encoding for categorical features
- Cyclic year features (<code>sin_year</code> and<code>cos_year</code>)
- 3 custom target transformations using Kaplan Meier and Nelson Aalen methods
- Classification and Regression models
- Models trained with entire train set, didn't use any cross-validation method
- Predictions were scaled before blending
