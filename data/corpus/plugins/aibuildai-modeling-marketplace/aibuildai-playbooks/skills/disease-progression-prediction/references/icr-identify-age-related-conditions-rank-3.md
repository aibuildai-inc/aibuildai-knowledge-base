# 3rd Place Solution for the "ICR - Identifying Age-Related Conditions" Competition

Competition: icr-identify-age-related-conditions
Rank: #3
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/430978

First of all, I was really surprised to be able to achieve this result. In fact, after participating in the competition and simply implementing a baseline code, I rarely paid attention to this competition again because the company usually worked overtime😅.
The final result was a baseline code based on the catboost model at that time, and its public score was 0.21.
Due to anonymous data features and medical-related health features, my initial idea is to construct new features through the ratio between different features, just as some indicators in the medical examination report are also calculated by the ratio between other indicators.
Before this, I planned to filter some anonymous features through corr, so as not to construct too many invalid features. However, without further attempts, the final code is still the cross calculation of all features.
The more effective operations in this competition should be the following two points
1.One is the cross calculation of features
2.The other is the catboost model
Because my lightgbm model with the same features got 0.22 on public score and 0.38 on private score.
here's my solution:
https://www.kaggle.com/code/junyang680/icr-lightgbmbaseline
Regarding the parameter selection of the lightgbm model and the catboost model, it seems to be based on the contents of some notebooks, but sorry, some have forgotten
