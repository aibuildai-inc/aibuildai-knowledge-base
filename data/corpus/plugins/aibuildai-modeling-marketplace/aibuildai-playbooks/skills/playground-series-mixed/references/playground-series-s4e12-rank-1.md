# 1st Place - Single Model - Feature Engineering

Competition: playground-series-s4e12
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s4e12/discussion/554328

This was a fun competition. In my first two playground competitions (Sept 2024, Nov 2024), feature engineering didn't improve CV nor LB too much, so in those competitions, I spent my time building a large ensemble of diverse models (GBDT, NN, SVM, etc). 

In this December Kaggle Insurance playground competition, feature engineering helped improve CV score and LB score, so in this competition, I was able to spend time building a single model and engineering features. It was very enjoyable to build a strong single model. My final submission is a single XGBoost model with 611 features! Thank you Kaggle for providing a fun competition that allowed Kagglers to practice categorical feature engineering!

# Solution Code Notebook
I published a simple version of my final submission [here][1] which achieves `CV = 1.019`. The full version model achieves `CV = 1.016` and takes 6 hours on 1xA100 GPU to feature engineer and train. The simple version takes 2 hours on 1xT4 GPU. The simple version uses only `229 out of 611 features`. It uses `learning_rate = 0.01` versus `0.001`, and `n_estimators = 2_000` versus `20_000`, and `target_encode(kfold=5)` versus `kfold=10`. (These simplifications shorten training time but decrease CV score and LB score).

# Feature Engineering Categorical Columns
A common way to improve the CV score and LB score of GBDT (gradient boosted decision trees like XGB, CAT, LGBM) is to provide various encodings for the categorical features. Given a categorical column, the basic encoding is `label encoding`. More advanced is `target encoding mean`(i.e. `TE`) and `count encoding`(i.e. `CE`). We can even `TE median`, `TE min`, `TE max`, `TE nunique`. We give the model the original column plus 6 different representations of the original. All 7 of these different encodings are input to the model and give GBDT multiple ways to understand the categorical column and improve CV score and LB score. And of course we can even invent more encodings.

# Create New Categorical Columns
Since encoding categorical columns improves our CV score and LB score, we can create more categorical columns by combining existing categorical columns. Then we can engineer more encodings from the new columns and improve CV score and LB score even more. In my published code [here][1], we create 20 new columns (by combining existing columns). We shared this idea in September's playground competition [here][2].

For example we can combine 2 columns together. If we have column `Occupation` which is categorical with 3 values `['Self-Employed', 'Employed', 'Unemployed']` and we have column `Gender` which is categorical with 2 values `['Female', 'Male']`. Then we can create a new column by combining these two with `train['new'] = train.Occupation +"_" + train.Gender`. Then the new column is categorical with 6 values `['Self-Employed_Female', 'Self-Employed_Male', 'Employed_Female', 'Employed_Male', 'Unemployed_Female', 'Unemployed_Male']`. We can also combine 3,4,5,6,etc columns together. 

# Treat Numerical Columns as Categorical
Another technique that often works is to treat numerical columns as if they are categorical. Then we can encode them with `TE mean`, `TE median`, `TE min`, `TE max`, `TE nunique`, and `CE`. And we can combine numerical columns with other numerical and/or categorical columns and apply `TE` and `CE` to the resultant new column. 

# Search Using GPU RAPIDS cuDF-Pandas!
In this competition after decomposing the `Policy Start Date` (into year, month, day, hour, etc), we have 23 original columns. If we create all combinations of 2,3,4,5 and 6 columns we will have 145_000 new columns! This is too many, therefore we need to find and use the best ones. 

For each combination, we need to create the new column, compute TE and CE which requires nested fold groupby aggreations, then train an XGBoost model. Evaluating a single combination takes time. Using [GPU cuDF-Pandas][3], we can search 10x to 100x more combinations than using CPU, wow!

During this competition, I left my computer running day and night in a for-loop running [GPU cuDF-Pandas][3] evaluating thousands of random combinations. Whenever feature engineering a combination with TE and CE improves CV score, the code saves it to a list. After running for multiple days, the code found 170 powerful combinations. I publish the best 20 combinations in my solution code [here][1].

# Learn More About Categorical Encodings
If you want to learn more about feature engineering categorical columns, NVIDIA [KGMON][5] will present a workshop at NVIDIA's 2025 GTC (i.e. GPU Technology Conference) in San Jose, California on March 17 thru 25th 2025 [here][4]. Specifically we will show how to perform TE (target encoding) and CE (count encoding) and explain why it works. And we will show how to incorporate  these features into models to  improve CV score and LB score! 

[1]: https://www.kaggle.com/code/cdeotte/first-place-single-model-cv-1-016-lb-1-016
[2]: https://www.kaggle.com/code/cdeotte/rapids-cuml-lasso-lb-0-72500-cv-0-72800
[3]: https://rapids.ai/cudf-pandas/
[4]: https://www.nvidia.com/gtc/
[5]: https://www.nvidia.com/en-us/ai-data-science/kaggle-grandmasters/
