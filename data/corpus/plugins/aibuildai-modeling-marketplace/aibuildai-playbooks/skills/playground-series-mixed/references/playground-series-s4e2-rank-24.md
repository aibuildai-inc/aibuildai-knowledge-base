# 24th Place Solution

Competition: playground-series-s4e2
Rank: #24
Source: https://www.kaggle.com/c/playground-series-s4e2/discussion/480927

Even though 24th place is not like top 3 or something, I am so happy to have accomplished this. I've always wanted to dedicate some time to Kaggle to enhance my skills and this time not only I managed to do it, but also got rewarded with 24th place. 

Ok, enough about my feelings, and let's go to the solution explanation.

## Main idea

The idea behind the solution was based on code from @divyam6969 [1].
It is pretty much an ensemble of XGBoost and LightGBM. I didn't add any features and used StdScaler for the numerical ones. For the categorical features, in XGBoost I used MEstimateEncoder (didn't know until this competition) and for the LightGBM I used a OneHotEncoder.

The CV consisted of a 10-split Stratified K-fold. The crucial part was tweaking the weights of the ensemble, which led to the difference in LB.

## What didn't work

- Additional features such as BMI 
- Catboost didn't contribute to the CV
- Other types of scalars and categorical encoding
- Pruning the dataset

## What I've learned
- The classic: Trust in your CV over LB
- Experimentation tracking (for this particular competition I used MLFlow and logged the important parameters and metrics)
- One can always look up to other solutions, be it to extend them or incorporate the ideas into your own solution.

## Kernel
https://www.kaggle.com/code/nicowxd/ps4e2-top-25-simple-ensemble-private-0-91067

## References used (Thanks to the authors)
[1] https://www.kaggle.com/code/divyam6969/best-solution-multiclass-obesity-prediction
[2] https://www.kaggle.com/code/ddosad/ps4e2-visual-eda-lgbm-obesity-risk
