# 19th Place : Solution Approach and how we got a huge boost in score!

Competition: playground-series-s3e10
Rank: #19
Source: https://www.kaggle.com/c/playground-series-s3e10/discussion/396471

Me and my teammate combined our solutions at the end only to receive a very huge boost in score!

The following notebook details my approach to this problem statement.



>My Solution Approach

- I basically used an ensemble of 3 models (XGB,CAT,LGBM) along with hyper-parameter tuning. Adding the original dataset didn't improve my score. But the interesting thing I noticed was, Tuning the models based on the original dataset boosted my score a lot on the CV and LB. 

- For hyper-parameter tuning, I used Optuna. I performed a 5 fold CV within the optuna to obtain the mean log loss for each selected hyper-parameters. This ensured that I was tuning based on my CV within the optuna function. 

- Finally, I combined the models using weighted average, I tried various weights combining approach like considering the CV score, Optuna based tuning and many more. But what worked was manual selection. I had to do an exhaustive search to get the best combination of weights. 


>What I tried but didn't work :

1. **2 Level stacking approach** :  It gave me the same CV score. So I didnt bother to complicate the model complexity and risk overfitting.

2. **Feature selection**: My idea was to fetch the most important feature (EK) and combine it with all other features and to check for the resulting accuracy. It improved my CV a bit but gave a bit dip in LB, So I didnt risk the difference.

3. **KNN Approach** : This was something unique I tried. Basically I got the predictions of the ensemble on the entire training data (Obviously using CV), and stored it separately in a data frame. During prediction, Once I got the predicted value using ensemble, I found the top N nearest predictions and tried averaging it with more weightage to the actual prediction. My intuition was that, If predictions are very similar, The data points must be very similar,. So I tried this approach but it didnt work out unfortunately. Probably it ended up adding more and more bias!



>How we got a huge Boost in Score :

Basically, Me and my teammate developed our own solutions separately. His approach was more focused towards feature engineering which worked well for him. Mine was focused more on model building stage. The diversity of our models when combined gave us a huge boost in the LB. 

**My partners solution thread** : https://www.kaggle.com/competitions/playground-series-s3e10/discussion/396259


*Hope I helped you guys gain some ideas for your next comp! :)*
