# Winner #1 Position Mini Writeup

Competition: santander-value-prediction-challenge
Rank: #1
Source: https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63907

Thanks Santander and Kaggle to make this competition possible and permit the competition continue even after the leak disclosure. That decision makes a very tough competition, since the real data becomes very small in the trainset.
As I'm attending KDD in London I'm not able to access my full repository of scripts to write a full solution description, so I decided to start with a mini write-up from things that I remember from my mind.
As I would like to thanks my team mate Lucasz who shown great skills in finding the groups of features and leaks. Lucasz will write about leak searching in another post.

We found 27477 rows in testset that are not being scored either in Public and Private. These rows are the ones that presents multiple nonzero values with more than 2 decimals. So I dropped these rows from all datasets. I suspect these rows are created artificially just to add noise to testset and avoid probing. Also it explains why my DAE models trained on both train+test are not working before I found it.

At the end we found 113 groups of 40 features each, totalizing 4520 features. 
Using those groups information makes possible to find 11784 label leaks: 3887 at train and 7897 at test. After some experiments we decided to use only strict leaks, in other words, only the leaks found directly without multiple candidates. So our hit rate in 100% for all leaks.

Using the leak mechanism, we also found groups of rows we believe are from the same Santander customers/users. Taking it into account there are only 513 users in train and 13857 in test. In train from that 513 users, 338 have groups of two or more rows and 175 are single row users. And for test there are 726 multiple rows users and 13131 single row users.
This is important because single row users have at the maximum 40 timestamps in history. But using multiple row information is possible to reconstruct the timeseries information and we found some users with up to 167 time stamps combining all row groups. 

So our solution is a blend of non-leaked models, replacing the predictions by the leak as a post processing step.

I tried several models for this competition, something around 100(I didn't count). But I choose some of then for the final solution. 

Before the leak my best blend of models is scoring around 1.36 in Public LB. Those models are built using features like histogram, density, max, min, sd and the original features. Histogram features makes a lot of sense, since everything is a time series in a black-box.  At this point XGBoost is performing better than LightGBM. Also I fixed 10 sets of 20-fold CV at that stage. So each model ran 200 folds and then average to decrease variance. 

After the leak I started to use columns and group information to build the features and it makes possible to improve performance of models to 1.24 LB without using the leak, but using the right dataset information and feature engineering. So I started training multiple versions of models each one exploring one characteristic of the dataset. But at this point I had to change the cross-validation strategy and I fixed only 1 set of 10-FOLDs stratified by group_id. This is very important to avoid leaking user information across folds and makes possible to build more stable models and tuning hyperparameters better.

Another good solution I remember was to fit one model for each group of features, but only for rows that have at least one nonzero element. So I trained 113 models that way. After that I stacked all 113 out of fold predictions and ran a second level fit using it. It scored around 1.285 CV and 1.27LB.

I will detail it better when I return to Brazil :-)

Giba
