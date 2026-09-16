# 28th Solution

Competition: icr-identify-age-related-conditions
Rank: #28
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/433013

To begin, I'd like to express my gratitude to Kaggle for hosting this competition, and extend my heartfelt congratulations to the winners of this competition!

Here is the main takeaway from my solution. After reviewing discussions on the dangers of post-processing and data distribution on the forums and conducting several experiments, I've come to the realization that I should emphasize the prevention of overfitting due to the small dataset, rather than complicating my model. I allocated a significant amount of time to feature engineering while focusing on maintaining simplicity in model building.

I used stratified cross-validation, and an XGBoost model was constructed with specific parameters obtained through RandomizedSearchCV. A lot of normal feature engineering methods were used. 

the public score for my submission is 0.18
