# 8th Place Solution for the "ICR - Identifying Age-Related Conditions" Competition

Competition: icr-identify-age-related-conditions
Rank: #8
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/430897

Frankly, I was somewhat shocked by the result. I had a pretty solid idea that there would be a lot of shakeup - but certainly wasn't expecting to be in the top 10. 

The public/private scores for this submission are 0.19/0.34 respectively.

# Context:
Business context:[ https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview)
Data context: [https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data)

# Overview of the approach
I focused first on predicting if someone had a specific Age-Related Condition rather than the class. This proved more effective than predicting class alone. Effectively, I create an Ensemble of Ensemble Predictors focused on specific conditions. The models in the primary ensembles were XGBoost & TabPFN. 

# Details of Submission

## Imputing Strategy
My primary imputing strategy utilized XGBoost to predict most of the missing values rather than dropping, filling with 0, mean, mode, etc.

Compared to median imputing, this resulted in an improvement in the Public score of 0.04 (with no change in Private Score). 

In this submission, there were two fields that contained the majority of the nan values in the training set. As I couldn't be sure this would hold true in any other set, I created a function to impute these values in a brief loop while dropping additional columns from subsequent impute runs when multiple columns in the target were nan values. 

# Context:
Business context:[ https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview)
Data context: [https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data)

# Overview of the approach
I focused first on predicting if someone had a specific Age-Related Condition rather than the class. This proved more effective than predicting class alone. Effectively, I create an Ensemble of Ensemble Predictors focused on specific conditions. The models in the primary ensembles were XGBoost & TabPFN. 

# Details of Submission

## Imputing Strategy
My primary imputing strategy utilized XGBoost to predict most of the missing values rather than dropping, filling with 0, mean, mode, etc.

Compared to median imputing, this resulted in an improvement in the Public score of 0.04 (with no change in Private Score). 

In this submission, there were two fields that contained the majority of the nan values in the training set. As I couldn't be sure this would hold true in any other set, I created a function to impute these values in a brief loop while dropping additional columns from subsequent impute runs when multiple columns in the target were nan values. 

```
# For Use with Float Fields
def imputeRegressor(df, field, dl=[]):
    print(f"Impute Regressor w/ XGBoost for {field}")
    print(f"NaN Values @ Start: {df[field].isna().sum().sum()}")
    # Multiple Fields May contain NA at the same time.
    drop_list = set(df.columns).intersection(set(dl))
    imputeDF = df.copy()
    imputeDF.drop(drop_list, axis=1, inplace=True)
    impute_X = imputeDF.drop(field, axis=1).dropna()
    impute_y = imputeDF[field]
    imputeDF = impute_X.join(impute_y)
    trainDF = imputeDF[imputeDF[field].notna()].dropna()
    train_X = pd.get_dummies(trainDF.drop(field, axis=1))
    train_y = trainDF[field]
    # We can only solve for the ones where other fields are not na.
    testDF = imputeDF[imputeDF[field].isna()]
    test_X = testDF.drop(field, axis=1)
    test_y = testDF[field]
    test_X.dropna(inplace=True)
    testDF = test_X.join(test_y)
    test_y = testDF[field]
    print(f"Predicting for a training set of {train_X.shape} and test set {test_X.shape}")
    if (len(test_X)) == 0:
        print ("Nothing to Predict")
        return
    # Handle Different Feature Counts
    colsToUse = set(train_X.columns).intersection(set(test_X.columns))
    train_X = train_X[colsToUse]
    test_X = test_X[colsToUse]
    modelOne = XGBRegressor(n_estimators=200).fit(train_X, train_y)
    preds = modelOne.predict(pd.get_dummies(test_X))
    predictedDF = test_X
    predictedDF[field] = preds.astype(float)
    # Save out the Results to the base DF
    for i, row in predictedDF.iterrows():
        df.loc[i, field] = row[field]
    remainingToSolve = df[df[field].isna()].count()
    print(f"NaN Values @ End: {df[field].isna().sum().sum()}")
```

## Feature Engineering
Aside from the imputing step above and converting categoricals into ints, I did no manipulation of the float data. There are quite a few fields with outliers that could perhaps be addressed, and at least one where the values appear to max out. In this case, I didn't have time to play around.

## Ensemble Models Used
XGBoost's Classifier w/o fine tuning (fine tuning overfitted)
TabPFN

As a metric (for XGBoost) I used balanced log loss, with the binary logistic objective. 

Each model was fit using cross validation, scored by Balanced Log Loss, and the best performing model from each run was chosen. 

## Greeks & Class
The greeks file contains the Alpha field that differentiates the specific conditions that might result in the Class value being equal to 1. The remainder of the Greeks file was ignored. 

## Ensemble of Ensemble Strategy
Instead of a single ensemble predicting Class, we train 4. We train ensemble models to predict

1. Class
2. Alpha "B"
3. Alpha "D"
4. Alpha "G"

For Positive predictions, we take the max of positive predictions for B,D,G & average the result with the positive prediction of class.
For Negative predictions, we take the min of negative predictions for B,D,G & average the result with the negative prediction of class.

Note that in the code below, I refer to the predictors for B, D, and G, as Beta, Delta, and Gamma. This is not to be confused with the fields of the same name (which are not used). 

```
def ensembler(train_X, modelAlpha, modelBeta, modelDelta, modelGamma, postProcess=False):
    """
    Creates an Ensemble Prediction for Train_X based on the respective features
    Ensembles the Alpha, Beta, Delta, & Gamma Predictors.
    Beta, Delta, and Gamma individually predict each disease while Alpha predicts overall disease. 
    We combine these together to generate an ensemble of the ensembles. 
    """
    alphaPreds = modelAlpha.predict_proba(train_X, False)
    betaPreds = modelBeta.predict_proba(train_X, False)
    deltaPreds = modelDelta.predict_proba(train_X, False)
    gammaPreds = modelGamma.predict_proba(train_X, False)
    ensemblePreds = []
    for i, pred in enumerate(betaPreds):
        truthy = mean([max([betaPreds[i][1], deltaPreds[i][1], gammaPreds[i][1]]), alphaPreds[i][1]])
        falsey = mean([min([betaPreds[i][0], deltaPreds[i][0], gammaPreds[i][0]]), alphaPreds[i][0]])
        ensemblePreds.append([falsey, truthy])
    # Post Process our Predictions
    if postProcess:
        ensemblePreds = modelAlpha.post_process_proba(ensemblePreds)
    return np.array(ensemblePreds)
```

### Special Note on Balanced Log Loss & The Greek Strategy
For this competition, we did not need to ensure that predicted probabilities sum to 1. There are quite a few public notebooks utilizing a version of this function that simply inverts the positive predictions. This would obviously be incorrect with my prediction strategy described above as I can have total probabilities both greater and less than 1. For this notebook, I chose to ensure the Balanced Log Loss function I used took the full prediction probabilities and dealt with them independently rather than ensuring the probabilities summed to 1. Both should presumably be equivalent. 

### Caveats from a better performing notebook.
A better performing notebook (Private 0.33) ignored the Class model altogether.

## Prediction Balancing
I balanced the prediction probabilities based on the weight of the predictions by the ensembled' models on a per model basis during training, and in aggregate when making predictions. I found this performed better during limited testing than providing an up-front weighting to XGBoost for the classes based on the ratio in the training set. 

## Picking Features
For each of the Ensemble Models (Class, B, G, D), I ran it through a loop removing features with feature importances scored <=0 to check which had the best performance through training. I kept the feature set with the best scoring run.

### Caveats from a better performing notebook.
A better performing notebook (Private 0.33) used all features.

## Hyperparameter Tuning
I ran through 200 trials w/ Optuna on each model (Class, B, D, G) to tune hyper parameters. Ultimately, I found that tuning resulted in worse scores for both public & private sets (0.01 to 0.05 worse). However, given the time limitation, I did not re-perform this for the reduced feature set for each model. This leaves room for re-running hyperparameter tuning on the reduced feature sets. I still suspect that at this point we're already working on overfitting against my core solution.

## A note on Post Processing
I tried simple prediction post processing strategies where I pushed high values towards 1, and low values towards 0. I also tried ratio oriented approaches. Both tended to improve the training result score significantly (~0.5 all the way down to ~0.04 range) - but this resulted in significantly worse scores on both the public and private data set. I observed this strategy in the highest scoring public notebook ([https://www.kaggle.com/code/vadimkamaev/postprocessin-ensemble](https://www.kaggle.com/code/vadimkamaev/postprocessin-ensemble)) that seemed to have been the source of many high scores. I'm curious now if this harmed those notebooks in a fashion as it did mine. 

# Sources
([https://www.kaggle.com/code/vadimkamaev/postprocessin-ensemble](https://www.kaggle.com/code/vadimkamaev/postprocessin-ensemble))
Inspired my failed attempts at post processing & source of my swap to balancing predictions during prediction rather than via the xgboost class weighting parameter (tabPFN didn't seem to have this, and attempts to mix/match were unnecessarily complex).
