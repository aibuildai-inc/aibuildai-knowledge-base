# 5th Place Solution | Learnings

Competition: playground-series-s4e4
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s4e4/discussion/499204

Hello All, 

This was a stable competition with less shake up because of the large volume of data and the metric RMSLE. I'm happy to share a note of my work in this competition. 

Firstly, the competition was very similar to [Crab Age Prediction](https://www.kaggle.com/competitions/playground-series-s3e16) , Interestingly I was Public LB 1 in that competition and fell to 40 in private. Hence I used the similar approach to estimate new features. 

I'd like to thank two public notebooks for their amazing work that helped me in this competition:
1) Notebook by [mfmfmf3](https://www.kaggle.com/code/mfmfmf3/clean-code-voting-regressor-base-3-models) , replacing the work with my feature engineering resulted in a really good score that was used to blend later. 
2) Notebook by [igorvolianiuk](https://www.kaggle.com/code/igorvolianiuk/abalone-rings-ensemble)

Many high scoring public notebooks were available but their score were a result of a blend from some other work and that is the only reason for not using them.  

### Approach

| Methodology | Details |
| --- | --- |
| New Features |  Top Surface Area, Water Loss, Measurement Ratios, Abalone Density, BMI |
| Feature Engineering | Discrete & Categorical Encoding, Transformations on Numerical Features |
| Models | XGBoost, CatBoost, LightGBM, ANNs |


### Learnings
1. 15 fold & 20 fold Cross validation increased my score than 5 fold. 
2. A lot of importance was given on tuning hyper-parameters and that worked well. 
3. I have improved a lot in using ANNs in ensemble modeling. 
4. I have only used **Harmonic Mean** to blend results, this is really important to give an edge in the private LB and avoid overfitting. 

Thank you all for the support & Wish you all the best in the next competition!

Happy Learning!
