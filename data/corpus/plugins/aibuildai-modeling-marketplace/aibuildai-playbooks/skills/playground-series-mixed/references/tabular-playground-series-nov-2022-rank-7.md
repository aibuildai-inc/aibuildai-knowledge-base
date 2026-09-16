# 7th place solution

Competition: tabular-playground-series-nov-2022
Rank: #7
Source: https://www.kaggle.com/c/tabular-playground-series-nov-2022/discussion/369731

Hi kagglers!

Thanks to Kaggle for the TPS competitions, it's the one I would like to participate since a long time.

## Team name
@mpware, solo

## Team place
97 public LB (0.51430) and 7th private LB (0.51892)

##The brief description of the solution

I've spent some time to find the best models among the 5000 provided. I've used two features selection method that gave me nice results on CV:

- **CatBoost FE based on LossFunctionChange**
    For each feature, LossFunctionChange represents the difference between the loss value of the model with this feature and without it.
    https://catboost.ai/en/docs/concepts/python-reference_catboost_select_features
    I've explored 1000, 500, 275, 250, 200 features. Best results was for 250.
	
- **Null importance FE with 100 runs**
    Shuffle the target randomly and compute features importance, run this process many times (100) to get importance distribution. 
	Compare with importance without shuffle (i.e. the problem we want to solve), if such importance is mixed within the ones with shuffle 
	then it means this feature does not matter for our problem. We keep only the ones that matters
    https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances
	At the end of this process I've selected 350 features

I've trained models based on these FE. Best CV (5 folds) was for classic LightGBM/GBDT, CatBoost and NN was not bad.

Finally my **submission is just a simple average** of:
- 3 LightGBM/GBDT models with 250, 275, 350 features
- 3 NN models with same 250, 275, 350 features
- 3 best single models: 2xLightGBM/GBDT(250 and 629 features) + 1xCatBoost (275 features)

To get an idea of the CV/LB variance/stability I've used 5 random seeds for the whole process. It was around +- 0.005 on both CV/LB. LB shake up risk might be high with such variance.


## Used libraries/algorithms
- CatBoost
- LightGBM/GBDT
- TensforFlow/Keras

*Note: I've tried AutoML (AutoGluon) tools for the first time.  It did not help to improve score but it gave quick reference results that help to go ahead in the right direction.*


## The key points which improve your score the most
- Features selection 
- NN + GBM ensembled models
Hyperparameters grid search result was not significant.


## Several ideas that doesn't work
- Shifting predictions by -+ 0.00XX
- LOFO for FE
- Drop features with hard samples (because they were already in my best list)
- All kind of L2 models were not significant


## Lessons learned from the competition
Trust your CV, play with random seeds to see how stable are your predictions.
 
Cheers.
