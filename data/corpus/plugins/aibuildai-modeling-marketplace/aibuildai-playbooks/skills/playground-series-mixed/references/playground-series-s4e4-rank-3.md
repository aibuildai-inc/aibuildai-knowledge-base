# 3rd Place Solution for the Regression with an Abalone Dataset Competition

Competition: playground-series-s4e4
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s4e4/discussion/499747

Thanks for holding this excellent machine learning game. I'm happy here to share my learning and experience during the match.

## **Context**
Business context: https://www.kaggle.com/competitions/playground-series-s4e4
Data context: https://www.kaggle.com/competitions/playground-series-s4e4/data


## **Overview of the approach**
Two key factors that contribute to higher position in the final leaderboard are:
* Feature engineering strategy
* Model ensemble way

For feature engineering, I applied a customized version of [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) in several of final sub-models. More details will be covered in later section.

The final model was a ensemble of 6 models as below:
* AutoGluon selected model with raw feature
* AutoGluon selected model with OpenFE* 30 features
* AutoGluon selected model with OpenFE* 30 features plus pseudo label data
* Ensemble of 3 sub-models(XGB,LGB,CAT) with Optuna tuned weights
* Ensemble of 3 sub-models(XGB,LGB,CAT) with Voting regressor
* Best ensemble of LGB models from [this post](https://www.kaggle.com/code/trupologhelper/ps4e4-lightgbm-only)

The NN ones published in "Code" forum is good but finally I decide to not include it as it doesn't perform significantly better than other candidates in local CV set.


## **Details of the submission**
### **Cross-validation is first key**
In this playground series, the public leaderboard is tested on approximately 20% of the test data while the private leaderboard is tested on approximately 80% test data. This usually lead to different performance on both boards. Hence, to build a reliable cross-validation way is almost first key in solutions.

### **Start with AutoFE and end with careful feature selection**
In feature engineering part, I made some modification on [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) for the initial feature engineering to handle the RMSLE metric. Of course, you could apply the np.log1p and np.expm1 to achieve the same effect.

After having initial feature candidates, it'd better to do careful feature selection because:
* reduce feature dimension to avoid overfitting
* avoid long training time
* make model more general to unseen samples

In my solution, I applied RFE (Recursive Feature Elimination) to select important features with a limited size (30 in my case) with Logistic Regression. The most important thing is to apply the feature selection process on a cross-validation way. If only applied on a single fold, the selected feature set could be not generalized enough on different data. It could be more stable by average the feature importance on multiple folds.

## **Ideas Tried But Not Working**
* Classification model instead of regression model
* More weight on original data instead of synthetic data

## **Some Suggestions To Beginners**
Everyone could have chance to win in the competition once you're well-prepared. Below are some general guidance that may help you improve the model performance:
* Data study: methods like feature correlation analysis, model error analysis could bring you new feature or ideas.
* Model tuning: even with same model, different hyper parameter may lead to huge performance gain.
* Selection and ensemble: trying different model combination is one direct way to improve the result.
* Learn from others: engage with Kaggle community, participate in forums and learn from others' posts or codes.

Wish all of you have a good journey in those competitions.
