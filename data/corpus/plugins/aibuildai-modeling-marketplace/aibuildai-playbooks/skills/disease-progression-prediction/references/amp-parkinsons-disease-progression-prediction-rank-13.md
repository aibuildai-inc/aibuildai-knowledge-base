# 13th place solution

Competition: amp-parkinsons-disease-progression-prediction
Rank: #13
Source: https://www.kaggle.com/c/amp-parkinsons-disease-progression-prediction/discussion/411436

First of all, congratulations to the winners!

My solution is based on LGB models, where I built a separate model for each forecast horizon (0, 6, 12, 24) and target (updrs_1, updrs_2, updrs_3, updrs_4).

The key variables I utilized for all targets were:
- "visit_month"
- "num_visits": Number of visits the patient had before the visit_month
- "relation" : Relation between these two variables 

The objective function was MAE. 

With these features, I achieved:
**CV ~ 54.57 (public score: 54.5, private score: 60.3)**

It is worth to say that about 70% of my submissions outperformed the two I eventually selected. (Luckily I got the gold)



**FEATURE SELECTION**

I performed an analysis to identify proteins that enhanced my local validation, particularly for updrs_1, updrs_2 and updrs_3. This was done by running multiple fold divisions and taking the average of CV to mitigate randomness.

**OTHERS THINGS**
- I introduced random noise to the protein NPX values to prevent overfitting. 
- I randomly set 15% of the protein NPX values to Null, also as a precaution against overfitting.
- I did target transformation for updrs_3 and updrs_4 (np.log1p/np.expm1)


With all this **My best local validation score was 53.46 (public score: 54.8, private score: 60.9).**

**THINGS THAT DIDNT WORK FOR ME**

- Remove outliers for training
- Customize objective function
- Ensembling with different models
- Predict trend and then predict residuals



**Lessons learned**
- I should have chosen one submission based on the public LB and another based on CV, instead of both relying solely on CV. 
- Next time, I will dedicate more time to conducting a deeper Exploratory Data Analysis (EDA)./
