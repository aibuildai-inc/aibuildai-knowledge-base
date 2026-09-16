# 12th place solution

Competition: porto-seguro-safe-driver-prediction
Rank: #12
Source: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44642

Thanks to Porto seguro and kaggle bring us such a wonderful competition, I learn a lot from this competition, I can feel the strong enthusiasm of the kaggle community, many people share their experience in actively, it is unbeatable,  it makes me grow up and gain valuable experience. It's so great.

**Final score**

We best private score :0.29187,  very lucky, the final result is not overfitting, and still stay in the top20.

**Data cleaning**

We removed all the columns that started with “ps_calc_”, and removed "ps_car_11", “ps_ind_11_bin”, these columns seems to be some noise.

**Feature engineering**

We didn't do a lot of complicated feature engineering, we just did OHE here, total is 207 features, we have been using these 207 features all the time.

**GBDT model(using R)**

 - XGBoost: without cv folds/207 features/fixed a random seed
 - LightGBM: with 5 cv folds/207 features/fixed a random seed for each fold

**Neural Network model(using Python)**

Many thanks to Joe Eddy, we simply modified his public nn kernel. 
We divided the input into 3 parts:reg, car and ind and designed a nn architecture to capture 
their inner relations. 
The idea came from entity embedding. Usually people use a weight matrix to represents the inner relations of a category variable. 
So we can use a weight matrix to capture the inner relations of the so-call 'main-class' such as reg, car and ind.

**Ensemble strategy**

We tried stacking, but the results were not ideal, in this time, we borrowed the harmonic averaging method.

鲲's topic:
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41658

Wiki:
https://en.wikipedia.org/wiki/Harmonic_mean

![hmean][1](1)
![hmean2][2](2)

 [1]: https://wikimedia.org/api/rest_v1/media/math/render/svg/753130a05a1fab890e5785924b5bdbb5f97c8b6a
  [2]: https://wikimedia.org/api/rest_v1/media/math/render/svg/4b0ece72e6275f3628f83d849bd56bf5347bb818

**Blending model**

 1. 2 XGBoost models in (1) 
 2. 91 LightGBM models in (1)
 3. XGBoost + LightGBM + NN models in (2), weight is : 0.4, 0.4, 0.2

Finally, thanks to my teammates @lessonnair, we can't get such a good grade without him.
