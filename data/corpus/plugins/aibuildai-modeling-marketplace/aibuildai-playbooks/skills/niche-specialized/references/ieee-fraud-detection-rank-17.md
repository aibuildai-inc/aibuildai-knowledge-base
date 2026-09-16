# 17th place solution summary - there is nothing special

Competition: ieee-fraud-detection
Rank: #17
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111696

First of all, I would like to express my sincere gratitude to Kaggle, IEEE-CIS and Vesta for hosting the competition.
This competition was really amazing and fun and I learned a lot.

## Retrospection
When I submitted my last submission at the last day of the competition, I checked that the public ranking was 32th place, and I fell asleep hoping that the shake-up wouldn't be too bad because the 32th place was already well enough for me.
When I woke up, I couldn't believe that my final ranking went up 15th and won a gold medal in 17th place.
In fact, I had a nightmare of winning a bronze medal barely because of shake-up. 😂 

I have two main goals when joining this competition.
One is to be a competition expert by winning a bronze medal or more.
The other was that I played as a team in the previous instant gratification competition, so this time I wanted to experience the competition alone, end-to-end.
It was enough to achieve these goals and I never expected winning a solo gold medal.

I know how hard it is to win a gold medal through the past few competitions, and I think a lot of other teams are better than me, so I doubt if I deserve a gold medal. All I can think is that I was really lucky in this competition.

As you can see the summary and code I shared below, what I did was just a good use of reading the public kernels and the discussions.

## Solution Summary

### Validation Scheme
I thought setting up a reliable validation scheme is the most crucial part. Many people used group k fold with month as a group or k fold without shuffle, but I felt uncomfortable for predicting the past with future data.

Instead, reading various kernels and discussions, my strategy was to get an idea from Chris( @cdeotte )'s discussion( https://www.kaggle.com/c/ieee-fraud-detection/discussion/108571#625045 ).
For 5 folds, I used the last 20% of the train data as the hold out, and measured the average auc for validation score and the average number of iterations with 3 random seeds to predict test data. When predicting test data, I used stratified 5 fold without early stopping.

After this validation scheme was established, the validation score and the public lb trend were almost perfectly correlated. I think this is the part where I did the best in this competition.

Below is the validation score, public lb, private lb trend after my validation scheme is set up.


### Pipeline
After establishing the validation scheme, I followed Konstantin( @kyakovlev )'s discussion( https://www.kaggle.com/c/ieee-fraud-detection/discussion/107697 ) and created a pipeline like this:
1. Feature engineering based on discussions, public kernels, or features I thought myself
2. Evaluate the model performance using validation score
3. For adversarial validation, filter out features using Roman( @nroman )'s covariate shift
4. Re-evaluate the model performance and get final validation score
5. Predict test data and submit

### Preprocessing
- Memory reduction : https://www.kaggle.com/alexeykupershtokh/safe-memory-reduction
- Filling C nans with 0 in test
- Filling card nans : https://www.kaggle.com/grazder/filling-card-nans

### Modeling
I used two models: LightGBM and CatBoost. Both are based on Konstantin( @kyakovlev )'s public kernel.

I used LightGBM until ensemble based on https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda

After reaching public lb 0.9592 with LightGBM, I created a CatBoost model for ensemble based on https://www.kaggle.com/kyakovlev/ieee-catboost-baseline-with-groupkfold-cv

The score of the final model is:
- LightGBM validation score : 0.94838, public lb : 0.959666, private lb : 0.937376
- CatBoost validatoin score : 0.95212, public lb : 0.960571, private lb : 0.936798

### Feature Engineering
- Frequency encoding
- Feature combining
   - card1\_addr1, ProductCD\_card1, TransactionAmt\_dist2, ProductCD\_TransactionAmt, ProductCD\_cents
- Finding uid
   - Hint from https://www.kaggle.com/akasyanama13/eda-what-s-behind-d-features
   - uid : combine of ProductCD, card1-6, addr1, D1-TransactionDay
- uid aggregation
   - TransactionAmt
   - dist1
   - D features
   - C features
   - Some identity features : id\_01, id\_02, id\_05, id\_06, id\_09, id\_14
   - Some V features : V258, V294, V306, V307, V308
- C, V features nan group pattern

### Feature Selection
- Covariate shift from https://www.kaggle.com/nroman/eda-for-cis-fraud-detection
- Drop all V features

### Ensemble
Final ensemble is LightGBM, CatBoost weighted geometric mean
- LightGBM, CatBoost Ensemble validatoin score : 0.95541, public lb : 0.963251, private lb : 0.940934

### Code Link
I wanted to share the final code with the Kaggle kernel, but because of the memory limitations of the Kaggel kernel, I failed to load the code, so I shared the code on the github.
- Preprocessing : https://nbviewer.jupyter.org/github/tmheo/IEEE-Fraud-Detection-17th-Place-Solution/blob/master/notebook/IEEE-Preprocessing-uid-memory-reduction.ipynb
- LightGBM : https://nbviewer.jupyter.org/github/tmheo/IEEE-Fraud-Detection-17th-Place-Solution/blob/master/notebook/IEEE-17th-Place-Solution-LightGBM.ipynb
- CatBoost, Ensemble : https://nbviewer.jupyter.org/github/tmheo/IEEE-Fraud-Detection-17th-Place-Solution/blob/master/notebook/IEEE-17th-Place-Solution-CatBoost-Ensemble.ipynb

As you can see my solution, there is nothing really special or new. I just followed Konstantin( @kyakovlev )'s kernel and discussion. It was something like standing on the shoulder of giants and reaching out a bit more. 

Thank you and congrats to all the winners!
