# 8th Place Solution | So simple!

Competition: playground-series-s4e4
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s4e4/discussion/499258

1. feature engineering
- https://www.kaggle.com/code/arunklenin/ps4e4-abalone-age-prediction-regression

2. training
- Using Autogluon with hyperparameter tuning
`hyperparameter_tune_kwargs = {
    'num_trials': 5,
    'scheduler' : 'local',
    'searcher': 'auto',
}`

- And training setting
`predictor = TabularPredictor(
    label='Rings',
    eval_metric='root_mean_squared_error',
    problem_type='regression',
)
predictor.fit(
    train,
    presets='best_quality',
    dynamic_stacking=False,
    num_stack_levels=2,
    excluded_model_types=['KNN', 'NN_TORCH', 'FASTAI'],
    time_limit=3600*24,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    keep_only_best=True,
)`

3. ensemble
- notebook from : https://www.kaggle.com/code/trupologhelper/ps4e4-lightgbm-only
- autogluon
