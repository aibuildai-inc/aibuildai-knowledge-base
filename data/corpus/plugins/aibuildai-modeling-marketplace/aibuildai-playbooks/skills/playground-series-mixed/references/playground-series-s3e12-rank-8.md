# #8 Solution

Competition: playground-series-s3e12
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s3e12/discussion/402416

Hey guys, it was probably luck too, but I placed in the top 10.
Here is what I've done:
- Feature engineering creating two new datasets ([notebook](https://www.kaggle.com/code/donatoriccio/ps312-8th-place-feature-engineering/) for details)
- On these new datasets I've created two stacking models by taking the OOF predictions [(notebook)](https://www.kaggle.com/code/donatoriccio/ps3e12-simple-stacking-starter-with-vecstack)
**Stack 1:** LGBM, Gradient Boost, CatBoost, Random Forest. 
**Stack 2**: KNN, Logistic Regression, XGB, AdaBoost, ExtraTrees
Both used Logistic Regression as level 1 model.
- Average between the predictions of the two stacks

In the end, it was a very complex model, but I think the main advantage was given by the feature engineering process. My best tip for this competition (and for every other one) is to **ignore the public LB score, focus on your CV score** . In my case it was a simple 10-fold CV.
