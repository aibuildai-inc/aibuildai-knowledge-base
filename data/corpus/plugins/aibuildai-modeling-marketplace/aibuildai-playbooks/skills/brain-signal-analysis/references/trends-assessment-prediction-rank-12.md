# 12th Place Solution

Competition: trends-assessment-prediction
Rank: #12
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/163000

Thank you to the host for organizing this competition and thank you to Kaggle for the computing resources.

Congratulations to all the winners and medal finishers. Thanks for sharing your solutions!

## Parcellation
My initial approach was to use 3D CNNs but I couldn't get a good performance out of them so I switched to a parcellation-based approach.

I wanted to use a parcellation with a moderate number of brain regions to keep the number of features manageable. I ended up using the [AAL](https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_atlas_aal.html) parcellation which has 116 regions.

## Age prediction
For each of the 3D maps, I extracted the mean intensity in each region (116 regions x 53 maps = 6148 features).

I used the loading, fnc and these maps-derived features to train a ridge model. I got a decent score for age prediction by just tuning the alpha parameter and two weight variables. The weight variables are used to scale down the fnc and maps-derived features.

Then I extracted the variance for each region and added them to the feature list. This led to a 0.0007 LB boost.

I tried using skew and kurtosis as features but this didn't improve the score. My final age prediction is a weighted average of 3 ridge models.

## Domain variables predictions
Unfortunately, these  maps-derived features didn't seem to help much in predicting the domain variables. I only trained a single model for each of the domain variables.

## Site effects
I built a site classifier using logistic regression and ridge models. I later stumbled upon the [Combat](https://github.com/Warvito/neurocombat_sklearn) harmonization method and I used it to adjust the features for site effects.
