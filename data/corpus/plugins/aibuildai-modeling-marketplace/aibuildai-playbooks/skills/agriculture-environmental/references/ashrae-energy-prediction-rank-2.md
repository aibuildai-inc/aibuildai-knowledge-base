# 2nd Place Solution

Competition: ashrae-energy-prediction
Rank: #2
Source: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123481

Our team finished 2nd on private LB (12th on public). The private LB is finally released officially.

So happy and pumped up for winning in the money for the first time on Kaggle (after 6 years). Santa has been kind this year :-)
Our team will be kind too and share our complete solution.

## Solution Architecture:


**XGB:** XGBoost
**LGBM:** LightGBM
**CB:** Catboost
**FFNN:** Feed-forward Neural Network

## Short version

- **Remove noise (Very important)**
- Very few and basic features (For stability)
- Optimize models for each site+meter (For site-specific patterns)
- Ensemble of XGBoost, LightGBM, CatBoost, NeuralNetwork (To reduce variance)
- **Postprocessing (Very critical)**
- Leak insertion (Sucks, but probably doesn't matter)

**Final Ensemble (approximate):** 30% XGB-bagging + 50% LGBM-bagging + 15% CB-bagging + 5% FFNN

Many variations of XGB, LGBM, CB were bagged: at site+meter level, at building+meter level, at building-type+meter level. Bagged XGB gave the best results among the boosting methods.

FFNN was used only for meter = 0.
It gave very poor results for other meters and didn't add value to ensemble.
Also, FFNN was very poor for site-14 so we didn't use it and hence that tile is missing from the models section in the architecture diagram :-)

Our solution was built heavily on @oleg90777 's base XGB/LGBM setup (which scores 1.04 on LB without leak) and our key points were cleaning the data and post-processing the predictions (validated on leaked data and LB). Read more about it [here](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123528).

The final ensemble scores almost the best on public LB, on leaked data as well as private LB, so hopefully it is robust and useful.

## Long version

### Pre-Processing
A lot of the low values of the target variable seem to be noise (as discussed multiple times in the forums, specifically for site-0) and removing these rows from the training data gives a good boost in score which has been done by several other competitors too.

It was the most time consuming task as we visualized and wrote code to remove these rows for each of the 1449 buildings manually. We could have used a set of heuristics but that is not optimal due to some edge cases so we just decided to spend few minutes on every building and remove the outliers.

### Feature Engineering
Due to the size of the dataset and difficulty in setting up a robust validation framework, we did not focus much on feature engineering, fearing it might not extrapolate cleanly to the test data. Instead we chose to ensemble as many different models as possible to capture more information and help the predictions to be stable across years.

Our models barely use any lag features or complex features. We have less than 30 features in our best single model. This was one of the major decisions taken at the beginning of our work. From past experience it is tricky to build good features without a reliable validation framework.

### Modelling
We bagged a bunch of boosting models XGB, LGBM, CB at various levels of data: Models for every site+meter, models for every building+meter, models for every building-type+meter and models using entire train data. It was very useful to build a separate model for each site so that the model could capture site-specific patterns and each site could be fitted with a different parameter set suitable for it. It also automatically solved for issues like timestamp alignment and feature measurement scale being different across sites so we didn't have to solve for them separately.

Ensembling models at different levels were useful to improve score. Just bagging with different seeds didn't help much.

Site-level FFNN was used only for meter = 0. Each site had a different NN architecture.
It gave very poor results for other meters and didn't add value to ensemble.
Also, FFNN was very poor for site-14 so we didn't use it and hence that tile is missing from the models section in the architecture diagram :-)

For tuning of all models, we used a combination of 4-fold and 5-fold CV on month from training data as well as validation on leaked data.

### Post-Processing
We have shared our post-processing experiments in another thread: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123528

Since we remove a lot of low value observations from training data, it artificially increases the mean of the target variable and hence the model's raw predictions on test data also has an inflated mean. Since RMSE is optimal at true mean value, reducing the mean of predictions of test data by a reducing factor helps bring it down to its true mean, thus improving score.

We tried a range of post-processing values and finally ended up using 0.8 - 0.85 for most models.

### Ensembling
Our best single type model was XGB but LGBM was very close and CB was not very bad either. All scored in the range of 1.04 - 1.06 on the public LB without leak.

Since FFNN was built only for meter = 0, we ensembled differently for every site+meter combination using a weighted average where the weights were determined using a combination of CV score, LB score, Leak score and intuition.

**Final Ensemble (approximate) for meter = 0:** 30% XGB-bagging + 50% LGBM-bagging + 15% CB-bagging + 5% FFNN
**Final Ensemble (approximate) for meters 1, 2, 3:** 30% XGB-bagging + 50% LGBM-bagging + 20% CB-bagging

The final ensemble scores almost the best on public LB, on leaked data as well as private LB, so hopefully it is robust and useful.

### Leak
We used leak data primarily for local validation and for inserting into the test data as many competitors did. We didn't use any leaks outside of sites 0, 1, 2, 4, 15.

Since our core models were at site+meter level, we didn't explore leveraging the leaked data as additional train data.

### Team
Shout out to my **cHaOs** team-mates @oleg90777 (one of the best team leaders I've worked with), @berserker408 and @isanton.

### Code
We will be happy to share our entire code if ASHRAE / Kaggle can confirm if we can. No timeline / commitment on this.

### Credits
* ASHRAE and Kaggle for hosting this competition.
* Competitors who scraped data and made it public. You are winners too.
* Kaggle admins for working hard to make the best out of the leak situation.
