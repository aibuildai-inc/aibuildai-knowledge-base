# 4th place approach

Competition: tabular-playground-series-jun-2022
Rank: #4
Source: https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334497

Hello everyone!
My solution was much simpler than the winners, but I am happy it gave me this position! 
I am relatively new to Kaggle and this is the first time I am in such a position. It feels surreal! Null imputation is usually just a passing task in most of the other assignments, I never thought I will spend countless hours over a month for this task, and get such a result at the end!

I began with a perusal of some discussion posts regarding null imputations and read up on Abhishek Thakur's null value imputation tutorial (also mentioned in one of the posts here). I then went ahead with a simple imputer with descriptive statistics based baseline submission, mostly the mean and the median. Then I used some ML models like LGBM, Catboost and XgBoost, just like many of us. My score at this stage was in the range of 0.90- 0.86. I then borrowed some concepts from some yester model ideas at work, using other columns in the table to predict a column with lots of nulls. Considering this, I tried an approach using a predictive model to predict the F_1 and F_3 columns. I used the mean and median values of these columns as well, making some submissions through this time, trying to figure out a strategy. I wish to extend special thanks to the notebook 'TPS2206 The na count of each record is critical!', this provided me a new perspective to this competition. My final approach to the columns in group F_4_ includes a simple dense neural network (without a callback, considering my hardware constraints) on Keras and Tensorflow. My final submission for columns in groups F_1 and F_3 includes a combination of linear models and the descriptive statistics for these features. I also finetuned one of my model candidates for F_4 from another notebook titled 'TPS Jun 2022 PyTorch Lightning with NA counts'.

I have learnt a lot from this unique experience. I learnt of the denoising autoencoder (from the winner's solution), learnt of the TabNet (from one of the submitted codes), used the iterative imputer for one of my baseline submission and learnt that even simple models produce great solutions if executed well! I had a pre-conceived notion that Kaggle was all about complex and even more complex algorithms, which I myself refuted! 

My own EDA notebook for this competition received quite a few upvotes, many thanks for that! Also, special thanks to the below notebooks that helped me reach here- 
1. TPS Jun 2022 PyTorch Lightning with NA counts
2. TPS2206 The na count of each record is critical!
3. 0.89 IterativeImputer + Optuna
4. TPS22Jun Tensorflow with NA counts

Wishing you all the best for the next TPS challenge, the first unsupervised model in this series! 

The best part of such competitions is that everyone is a winner! I think the knowledge I gained from this assignment will surely help me at work and otherwise.
