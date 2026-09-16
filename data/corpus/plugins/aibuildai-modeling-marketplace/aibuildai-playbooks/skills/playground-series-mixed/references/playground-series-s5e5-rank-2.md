# 2nd place - Trust CV and diversity

Competition: playground-series-s5e5
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582700

The higher of my 2 selected subs was a rather stereotypical ensemble: 74 OOFs ensembled with Ridge. But when I looked back through my submissions, the highest one on the private LB was an 11 model ensemble, ensembled with HC (positive weights only) and it contained the following:
- 2 Autogluons, 1 without FE and 1 with autofeat features (thanks @masayakawamata !)
- 2 Catboosts without FE
- 1 LGBM with TE features
- 1 Catboost with TE features
- 1 LGBM goss with huber loss, no FE
- 1 Linear Regression model based on [this](https://www.kaggle.com/code/angelosmar1/s5e5-linear-regression-cv-0-05992) notebook (thanks @angelosmar1 !)
- 1 [ResMLP](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/578953) (thanks again @masayakawamata !)
- 1 [LNN](https://www.kaggle.com/code/nikitamanaenkov/predict-calorie-lnn) (thanks @nikitamanaenkov !)
- This one's my favorite (HC agreed with me :D), it is a Catboost trained on the original features + Catboost classifier probs (after splitting the target into bins), then I train another Catboost on the residuals (idea from [here](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/537202), thanks @cdeotte !)
