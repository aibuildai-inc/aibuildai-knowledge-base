# 8th Place Solution

Competition: march-machine-learning-mania-2023
Rank: #8
Source: https://www.kaggle.com/c/march-machine-learning-mania-2023/discussion/400834

Firstly, I would like to thank Raddar for his awesome public codes, as well as to Kaggle for hosting this great competition and providing an awesome platform. I have also learned a lot from other fellow Kagglers, so thank you all!


**Approach**
This was my third year participating in the March Madness competition. Based on my prior experience, I decided to take a conservative approach through an ensemble of various models.


**Data Preparation**
For feature selection, I used the features from Raddar’s public notebook. I created two sets of features: one with Team Quality features and the other without. I removed Team Quality features for some models.


**Modeling**
I used five different models: LGBM, Logistic Regression, CatBoost, XGB, and SVM. However, I dropped SVM as it was too slow to train. I combined the outputs of the remaining four models equally. While I initially used cross-validation to measure each model’s performance and used a weighted average of the output of each model in the prior year’s competition, I found that this approach did not work well. Therefore, I decided to use equal weighting to avoid overfitting this year.


**Post-Processing**
In the past, overriding the whole seeds 1-4 and 13-16 did not work well for me. So, this year I only overrode seeds 1-2 and 14-15. Additionally, instead of giving 99% probability for seeds 1 and 2, I took a conservative approach by giving 95% and 90% probability, respectively.


Overall, I believe that this approach with the tremendous blessings from God helped me win my first prize and gold medal in Kaggle after 5 years!

Code link: [https://www.kaggle.com/code/pwh70411/8th-place-solution](url)
