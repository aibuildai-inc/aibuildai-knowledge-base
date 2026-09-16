# 3rd private 6th public | Brute-force ensemble| Post-processing

Competition: playground-series-s3e16
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e16/discussion/416783

Hello all,

Thanks to Kaggle for the challenge/ episode! It was a great experience to participate in a good regression challenge in this episode. Also, thanks to the fellow competitors for making this a memorable experience.
My approach is as below-

**Data engineering**
1. I considered the data generation notebook provided [here](https://www.kaggle.com/code/inversion/make-synthetic-crab-age-data) as a base and generated a number of samples with varying parameters. I appended this additional synthetic data with the original data and used it collectively to train the model
2. I used the below additional features in the model otherwise. These features are mentioned in the below discussion post- https://www.kaggle.com/competitions/playground-series-s3e16/discussion/415721, many thanks to @pandeyg0811 for the efforts.
a. Meat yield
b. Surface Area
c. Weight/ Shuck Weight
d. Pseudo BMI
e. Weight/ Length Squared
f. Viscera Ratio
3. I also used log(1+x) transform on the weight column, but it did not help me greatly

**Model strategy**
1. I used a regression model approach rather than a classification approach using a stratified 5-fold CV. Validation was performed on the competition data only. CV correlated well with the public LB
2. I trained several candidate models with feature subsets, using the provided features and additional feature-subsets, from 6-10 features per model run. I trained several candidates using the same OOF structure, but different features per run.
3. I used tree models as alongside- XgBoost, LightGBM, Catboost, Gradient Boosting Machine, HistGradientBoostingRegressor as base learners
4. I used optuna and LAD regression to tune the predictions in a subsequent ensemble per run with(out) post-processing and discovered the favorable impact of post-processing (rounding to the nearest integer) on the CV score and LB score. Post-processing my predictions helped me improve my LB score considerably throughout the challenge
5. I did not post-process any predictions based on the training/ original data. Most of my models posited predictions in the range of 4-20 years only. I did not impute/ overlay any train/ original predictions on this data. Whenever I tried to do this, my score would decrease. Perhaps my method was plain and sub-optimal, while some creativity was needed herewith.

**What did not work**
1. TabNet regressor
2. Random Forest
3. Extra Tree
4. Linear approaches like Ridge/ LASSO

**What I could have done better**
1. I could have perhaps selected a better submission as my final submission. My best private set submission could be yielded a 2nd place in the challenge. It fared slightly worse on the public LB but had a better CV score. I choose the submission faring the best on the public LB as a candidate
2. I could have tried to perhaps use the original/ train data to impute edge cases better. I did not find sufficient time to do this creatively in the final days of the challenge, but I think this could have improved my score. I may try and do this later as a personal experiment. 

**My learnings**
1. Trust the CV
2. Never deviate from rule 1
3. Devise a good pipeline from Day1 to assist one throughout the challenge and beyond. Messy code structure seldom helps one across several challenges 
4. Fail quickly- conduct several quick experiments and devise a strategy in the initial days
5. Include features that add value and stay away from value-destroying features. I got a first hand experience of this point from this assignment to great effect
6. Submissions on the final day may not be the best overall. I experienced this truly well in the episode. My final submissions were created much before the final day.

Wishing everyone the best, happy learning and regards! See you in the next episode!
