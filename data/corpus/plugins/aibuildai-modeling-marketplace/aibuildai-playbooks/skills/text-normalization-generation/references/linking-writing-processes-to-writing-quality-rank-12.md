# [14th place solution] Keras ensemble

Competition: linking-writing-processes-to-writing-quality
Rank: #12
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466993

First of all, I would like to thank the organizers for organizing this competition.
Thanks also to all the kagglers who shared information in notebooks and discussions.
My model could not score better than the public notebook, but by ensemble I was able to improve the public notebook's score a bit.
This is my first medal and I am very happy!


# Preprocessing
I did some data cleaning.
For example, during holding down SHIFT, even if nothing changed, it seemed be counted as a new event about every 30ms, so combined them into one.


# Feature engineering(similar to the public notebook)
I created 69 features from essay, 90 features from train_logs aggregation and binning on idol_time and essay-related columns.
(really appreciate [@kawaiicoderuwu](https://www.kaggle.com/kawaiicoderuwu) for shareing [essay constructor](https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor))
I used nltk.tokenize to split the essay and then addressed grammatical errors such as sentences connected or single symbol left.


# Models
ensemble of LightGBM, XGBoost and Keras
10-KFold, tuned with optuna
(single LightGBM | CV: 0.597, Public LB: 0.593, Private LB: 0.574)
(single XGBoost | CV: 0.597, didn't submit)


# Keras model
simple 3-4 layers model
Score was scaled between (-1, 1) and output by softsign or tanh.
Performance of single keras was poor, but ensemble with GBDT model improved LB scores.
(single keras | CV: 0.620, Public LB: 0.605, Private LB: 0.577)
Stacking and classification did not work.
Since few people were using keras, it was probably just luck, probably not originally suited for this competition.


# Ensemble 
I used the following notebook for ensemble.
Thanks so much for sharing!
[LGBM (X2) + NN + Fusion](https://www.kaggle.com/code/kononenko/lgbm-x2-nn-fusion) by [OLEKSIY KONONENKO](https://www.kaggle.com/kononenko)
[Writing Quality(fusion_notebook)](https://www.kaggle.com/code/yunsuxiaozi/writing-quality-fusion-notebook) by [YUNSUXIAOZI](https://www.kaggle.com/yunsuxiaozi)
[LGBM (X2) + NN](https://www.kaggle.com/code/cody11null/lgbm-x2-nn) by [CODY_NULL](https://www.kaggle.com/cody11null)


Finally I blended my model and (LGBM (X2) + NN + Fusion) in 3:7 or 4:6 ratio.
My final Private LB Score was 0.564197, blended (keras_softsign * 0.25) + (keras_tanh * 0.05) + (LGBM (X2) + NN + Fusion * 0.7)

#  
Thank you to everyone who shared information in notebook and discussions!
Thank you for reading!
