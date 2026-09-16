# 4th solution, Congrats Balyor !

Competition: ncaam-march-mania-2021
Rank: #4
Source: https://www.kaggle.com/c/ncaam-march-mania-2021/discussion/230946

First, congratulations on the Baylor victory.

■**Acknowledgements**
Thanks to Kaggle and hosts for holding this competition.
Thanks Kaggler!! This is my first gold medal :)

■ **Result**
 4th（score 0.56346）

■ **Preprocess,FE**
-  I used the average of all system ranks and converted to Rating.
- I mainly used regular season stats.
+α
- TS％（True Shooting percentage)
- Four-Factor（eFG%,TO%,FTR,ORB%）
- PPP(Point Per Possession)
- POSS(Possessions)

■ **Model**
LightGBM only.
(NN not work.)

■ **CV**
CV used 2003-2019 for training data. (Excluding validation season)
I used one season for validation.(validation seasons are 2016,2017,2018,2019)

- Model 1 is Training 2003〜2015,2017〜2019, validation 2016. Valid score is 0.56796.
- Model 2 is Training 2003〜2016,2018〜2019, validation 2017. Valid score is 0.552339.
- Model 3 is Training 2003〜2017,2019, validation 2018. Valid score is 0.585605.
- Model 4 is Training 2003〜2018, validation 2019. Valid score is 0.481603.
 
■ **Prediction**
I submitted the average probabilities of the four models.
