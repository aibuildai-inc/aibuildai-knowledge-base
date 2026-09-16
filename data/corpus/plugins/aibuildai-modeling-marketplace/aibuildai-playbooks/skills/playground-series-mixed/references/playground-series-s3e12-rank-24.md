# #24 solution – Xdata + XGB + LGBM + CAT + Opt roc curve

Competition: playground-series-s3e12
Rank: #24
Source: https://www.kaggle.com/c/playground-series-s3e12/discussion/402398

Here comes a short summary of the 24th place solution.

**Data**

Used the competition data + the extra org. data available with no reduction, with only scaling as Feature Engineering.

**Models**

As it was not an imbalanced target, I used 8 folds training and added LGBM, XGB and CATBoost as models with some before good known parameters for similar problem. Used balanced and weighted settings of the target for the models, to get the target perfect balanced.

After every finished trained model per fold it also did a calibration of the roc curve based on the validation set before predict the probability.

**Postprocessing**

Run a optimizing of the best weighted value for the three models and also optimizing of the best power average value.

**Conclusions**

Noticed a very low validation score in one of the folds. With some more analyzing maybe cleaning and filtering of that data could help the final score.

I also tried some other massive feature engineering and models but I picked the solution based in the CV score not the public, it ended up to be a good choice.

That’s it!
Happy Kaggling! 😊
