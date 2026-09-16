# GIBA - #7 Place Solution

Competition: allstate-claims-severity
Rank: #7
Source: https://www.kaggle.com/c/allstate-claims-severity/discussion/26537

#7 Place Solution

Hey guys, that was a fun competition. Thanks to all organizers. Congrats to the winners and other top 10 competitors!!!

My solution is very simple and I didn't spent so much time in this comp.
I have basically only 4 base models trained on different target transformations and same 10 folds: XGB, Keras, SVR and libFM.

1)- XGB: I trained some XGBs adding some 2-way features interactios and on different target transformations like:  t^1, t^(1/2), t^(1/4), t^(1/8), log(t), log(t+100), log(t+200), log(t+400), 10/t. CV varies from 1123.2 to 1135. I used maxdepth no more than 10. Even a maxdepth of 7 can build good models. I changed a little the other parameters according the transformation used.

2)- Keras NN: Same as XGB I trained on different target transformations and bagged 15 times each fold: t^1, t^(1/3), t^(1/5), t^(1/9), log(t+50), log(t+150), log(t+300), log(t+500), 100/t.  CV varies from 1128 to 1145. Number of neurons varies from 32 to 512 per layer.

3)- Only 1 SVR model adapted from a Kernel to run on my folds. It took 3 days to run on all folds in parallel. Scored about 1159 CV.  C=1

4)- Trained two libFM models. First one using raw data. Second using all features as categorical. Scored CV around 1190 and 1185. 


To blend all models I used a simple scipy minimize function with Nelder-Mead solver. I used meta features from 1st level prediction as input and searched to the best weights that directly minimize the MAE function. It worked very well with CV and public LB, but overfited a little Private LB. My CV using that aproach was around 1116.2.
Also I used another trick to improve a little more. I optimized the models weights for different predictions ranges, for example. I run minimize for predictions range [0..2000] then other for [2000..]. 
Blending preditions of global optimizer with by range optimizer improved CV to 1115.80 and that is my final model.

I quit the competition basically 3 days before the end to have some vacations and I still on it. On my last competition day I found that using R gbm with laplace loss builds a second level model that scores around CV 1117.3. I'm sure blending that GBM model and my optimizer solution could improve a little more the results. Maybe a second level NNet can improve results also.

What I found is that feature interactions, bagging results and correct hyperparameter tunning is the best way to improve models for that noise dataset and MAE metric.

See you at Kaggle,

Giba
