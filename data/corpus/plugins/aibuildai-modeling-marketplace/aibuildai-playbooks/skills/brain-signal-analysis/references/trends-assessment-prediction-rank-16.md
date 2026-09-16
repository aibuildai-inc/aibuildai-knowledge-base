# 16th Place “Simple” Solution

Competition: trends-assessment-prediction
Rank: #16
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/163277

First of all, We([@Kanna Hashimoto](https://www.kaggle.com/kannahashimoto) , [@hatry](https://www.kaggle.com/toseihatori)) would like to thank you all for organizing a very interesting themed competition. 

# Solution overview
- Adversarial Validation for site1/site2
    - Our policy was that we should only model using features that do not affect the site's classification.(It may have been meaningless as a result.)
    - We did adversarial validation to classify site1(all training data) or site2(test data , which is clearly revealed to be site2)  for only loading features. It result AUC 0.9.
    - Excluding the top 3 features that contribute to the classification of site1 and site2.As a result, AUC decreased by 0.6 ~ 0.7.
- stacking
    - local cv of 1st models are 0.163 ~ 0.1575
    - 2nd model local cv is 0.15674, LB is 0.15766, private is 0.15786.
- We did not use 3d CNN models.

# features
- loading, fnc features
     -  We created multiplication and subtraction variables.
- Correlation between images
    - We used the correlation coefficients for the 53 fMRI 3d images for each user ID. Removing the 0 pixels improved the local cv a bit.
- histogram features, statistical features

    - Image histograms and statistics were also useful. Just computing histograms and statistics on the whole image improved the local cv, but features that computed on the images that had been split into smaller cubes were more effective.

# Models
- 1st models
    - Ridge + Optuna + random seed averaging
    - MLP + random seed averaging
    - LightGBM + random seed averaging
- 2nd model
    - Baysian Ridge + Optuna
