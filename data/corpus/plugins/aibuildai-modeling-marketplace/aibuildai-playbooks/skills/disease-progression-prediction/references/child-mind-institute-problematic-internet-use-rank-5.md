# 5th Place Solution

Competition: child-mind-institute-problematic-internet-use
Rank: #5
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552656

this is my approach for this competition:




1- **For Time Series:**
To make some useful features from the time series, I used clustering. I picked 6 parquet files (2 most variant files from each class), combined them, and fit 15 clusters using the KMeans algorithm. then i used this fitted model to extract cluster from other parquets
I assumed each cluster could represent a specific movement in the parquet data. Then, I averaged each cluster's values over the time duration and used that average as a feature for each user. So, in the end, I had 15 features per user, which represented the average of some kind of activity during the time the user wore the watch.
I was late in the competition and didn’t have enough time to clean the data or extract more useful features from the time series. I believe there was a lot of room to extract much better features from it.

2- **Choosing the Right Target:**
After some initial submissions, I found that using threshold optimization led to overfitting and inconsistent results. So, I ignored optimization and used PCIAT-PCIAT_Total as the target labels, and applied fixed thresholds [31, 50, 80]. Since the exact values between these thresholds were less important than their relative position to the thresholds for the final prediction, I binned the PCIAT-PCIAT_Total into 10 bins and adjusted the thresholds accordingly.

3 - **For Unlabeled Data:**
I used pseudo-labeling. I trained three GBDT models (CatBoost, LGBM, XGB), a Lasso regression model, and a neural network (256-128-64 architecture). Then I ensembled these models to predict labels for the unlabeled data. These new labels were then used to train my final models.

4-**For Final Prediction:**
For the final prediction, I used the same four models I had used in pseudo-labeling, trained on combination of original and pseudo labels. I trained the models on the entire dataset and validated only on the original labels. This approach gave me the following results for my winning submission (I didn’t use time series for this prediction):



Then, I trained GBDT models on the whole dataset (tabular data + time series features). And used all these models for final ensemble 

Combining these  models gave me a CV of 4.6 and a private LB of 4.81, which put me in second place. However, since QWK is a noisy metric, I tried adding some other models (logistic regression and decision tree regression) with lower CV scores. This improved my final CV and public LB slightly, but it dropped my private score from 4.81 to 4.77. I couldn’t blame for this because at the end the CV and LB and also the consistency between these two are the only metrics to choose final submission. 

special Thanks to  organizers and Kaggle and all participants who shared their knowledge. Best of luck in the next competition, and Happy New Year!
