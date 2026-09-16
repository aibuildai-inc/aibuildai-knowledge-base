# 24th Place Solution

Competition: equity-post-HCT-survival-predictions
Rank: #24
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566677

First, congratulations to the competition winners, and sincere thanks to our team members @gentlezdh,  @yoshifumimiya,  @lizhecheng,  @nandodmelo,  @kekshibata , as well as the competition organizers.
We present a short summary of our solution (**24th place, private LB 0.694**). More detailed version is [here](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/566809).

# Overview
We trained regressors and classifiers separately and combined their outputs using ensembling & post-processing. And the “Trust CV” strategy led us to a final LB score of 0.694 to win silver.


# Model Training
## Regressors:
- GBDTs (CAT, LGB, XGB) and AutoGluon on 8 targets with basic feature engineering.
- For specific 4 targets, applied extensive feature engineering with CatBoost and AutoGluon
- PRL NNs, including customized losses

## Classifiers:
- GBDT (CAT, LGB, XGB) classifiers
- Stacked logistic regression and combined Autogluon classifier.

# Ensemble & Post-Processing
- Combined regressors’ outputs using a **multi-task Elastic Net** on 8 targets. (excluding some custom PRL NNs as inputs).
- Post-processing: Adjusted predictions based on classifier probabilities with race-specific parameters optimized by Optuna.
- Blended in custom PRL NNs for high efs probability data and weaker races.

# Submission Strategy
- Best CV:  CV 0.6908 / public LB 0.690 / private LB 0.694
- Best LB:  CV 0.6853 / public LB 0.694 / private LB 0.692

Selected the best CV and LB for final submissions. “Trust CV” approach ultimately secured us the silver medal.
