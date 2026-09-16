# 9th Place Solution ( XGBoost+LGBM+NN )

Competition: amex-default-prediction
Rank: #9
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/350538

Many thanks to AMEX,Kaggle and all contributors of discussion during the entire competition ( @raddar , @cdeotte , @ragnar123 ,…). Congratulations to all winners and new Experts, Masters and Grandmaster!
## Score & Result
My best submission
CV: 0.799106        Public: 0.80062        Private:0.80875
My result
CV: 0.799194        Public: 0.80057        Private:0.80868
# Feature Engineering
I only used the integer dataset provided by @raddar. 
- Base features
aggregated features like mean,max,min,std,sum,medium,last,first
- other rate and diff features with datediff
last-first,last1-last2,last1-last3,last-mean,max/last,sum/last  and so on
- Date features
 is it a holiday 
# Model
- LightGBM 
3 models of LGBM with different data representation & parameters give CV in the range [0.796-0.799] and LB in [0.797-0.799]  (2 model with dart-LGBM , 1 model with goss-LGBM)
- XGBoost
6 models of XGB, with different data representation & parameters give CV in the range [0.794-0.796] and LB in [0.795-0.796]
-  NeuralNet
4 models  of NeuralNet  with different parameters give CV in the range
[0.788-0.790] and LB in [0.790-0.792]
(I am not so proud of NNs  Thank again @cdeotte for sharing his great public kernel NN.)
# Ensemble(stacking)
Using 13 models to stack with 10-fold cross-validation , Hyperparameter-tuning and appropriate early stopping can give   Private in the range [0.80853-0.80875] 
# some ideas
Predict if a customer will default  when customers have already used credit to consume ,trending of the features changed over time(like consumption frequency ,Change in consumption amount) is very important ,especially in the last few months.

I am very grateful to this competition. I learned a lot in this competition for a newbie in kaggle .Thanks you all😎
