# 6th Place Solution

Competition: playground-series-s5e1
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s5e1/discussion/560653

First of all, thanks to everyone who shared interesting discussions—I learned a lot from them!

My final solution consisted of an ensemble of 3 different models. The first model I used was an adaptation of @kdmitrie's published notebook. The second model used a multiplicative linear regression model, and the third model was a transformer published by @cdeotte.

## Model 1 (Public LB = 0.04792, Private LB = 0.04678)
- Multiplicative model
- Accounted for leap years (2012 and 2016). In practice, this just meant that I changed the dayofyear column to subtract 1 from all dates after the 29th of February for leap years.
- Used different country weights: I realized that the predictions for Kenya greatly influence the final score due to the nature of the MAPE metric (Try adding just 1 to all predictions for Kenya and see what happens!). I used a multiplier of 1.02 for Kenya on top of the multiplier of 1.06 for all predictions. I also experimented with slightly lower multipliers for other countries and used observation-specific multipliers where each multiplier was based on the confidence of the prediction (based on the variability of the same prediction in previous years).

## Model 2 (Public LB = 0.04874, Private LB = 0.04800)
- A linear regression model is trained which predicts the fraction of yearly summed num_sold for each day in the year. 
- The store fraction was obtained by just taking the mean fraction of num_sold within each product on a given day and country. 
- Product fractions (this means the product fraction of the total num_sold within a country in a day) were obtained using linear regression. Simple cos and sin transformations were used to make strong predictions here.
- I found it useful to first leave out the year 2016 for validation to investigate any patterns in the error.

## Model 3 (Public LB = 0.0526, Private LB = 0.06037)
- The third model was simply the output of the Transformer published by @cdeotte. 

## Ensemble of models (Public LB = 0.04706, Private LB = 0.04722)
- This was the ensemble of the three models.
- In hindsight, just submitting the predictions from Model 1 would have been better (easy to say now)!

# Things I tried that didn't work
- Using boosting methods (e.g., Catboost, XGBoost) and RNNs.
- Accounting for holidays in linear regression models (e.g., accounting for the effect after Easter).
