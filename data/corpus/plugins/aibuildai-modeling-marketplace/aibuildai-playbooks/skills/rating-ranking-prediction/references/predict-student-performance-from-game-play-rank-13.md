# 13th place solution

Competition: predict-student-performance-from-game-play
Rank: #13
Source: https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420077

# 13th place solution
First of all, I would like to thank the Kaggle community for sharing great ideas and engaging discussions. I would also like to thank the hosts for organizing this interesting task competition.

## Summary
- Ensemble of LightGBM and NN
- Cross validation: Nested cross validation
    - Training data: Data for which the first four digits of the session_id are 2200 or less.
    - Validation data: Data for which the first four digits of the session_id are 2201 or more.
    - I trained the model on the training data using a 5-fold cross validation strategy, and evaluated it on the validation data using predictions from all 5 trained models.
    - For the final submission, I trained the model on the entire dataset using a 5-fold cross validation approach.

## LightGBM
- I trained one model for level group 0-4 and another for level group 5-12. For these, I included features representing the target and trained a single model.
- For level group 13-22, I trained a distinct model for each target.
- Main features:
    - The count of categorical data for each session_id
    - The statistical measures of numerical data for each session_id
    - An aggregate of the next action taken 
- Scores
    - CV : 0.7032
    - Public Score : 0.704
    - Private Score : 0.701


## NN
- Model: Transformer + GRU
    - The standalone Transformer didn't perform very well.
    - The addition of GRU improved the score.
- Trained with fewer features
- I trained a separate model for each level group.
- Scores
    - CV : 0.7010
    - Public Score: 0.700
    - Private Score: 0.700

## Ensemble
    - LightGBM * 0.66 + NN * 0.34
    - CV : 0.7053
    - Public Score : 0.706
    - Private Score : 0.702
