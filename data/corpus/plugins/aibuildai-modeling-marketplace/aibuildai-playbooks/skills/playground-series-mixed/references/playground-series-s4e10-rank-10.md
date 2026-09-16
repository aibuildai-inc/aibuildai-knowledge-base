# 10th place solution : no blind blend

Competition: playground-series-s4e10
Rank: #10
Source: https://www.kaggle.com/c/playground-series-s4e10/discussion/543735

Hi,

Thank you Kaggle for this competition, congratulations to everyone and thank you to have shared so many usefull insights during this episode.

I'm on vacation a few days and write this message with my phone.

My solution is a **LogisticRegression** of 4 meta learners : each meta learner (bold below) is a stack of GBMs :



Boxplots are the 4 repetitions with various OOF predictions obtained with various random seeds : I wanted robust results.

Before this Logistic, I trained more than 30 GBMs, I tried everything I was able to try with categorical hyperparameters of XGBoost, CatBoost and LightGBM and I learned a lot from these personal experiments.

I kept both categorical and numerical features for several columns of train dataset (```person_income``` especially) . I didn't impute missing values, I didn't made feature engineering and kept the original dataset only for training (not for validation). I used optuna to fit hyperparameters of each GBM.

Here is my [final submission](https://www.kaggle.com/code/adaubas/pss4e10-logistic-of-gbms).

Good luck for next competitions and have fun !
