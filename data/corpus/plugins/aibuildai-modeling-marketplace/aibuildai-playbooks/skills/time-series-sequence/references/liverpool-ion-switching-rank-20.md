# 20th place solution [Single Model]

Competition: liverpool-ion-switching
Rank: #20
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153849

I use [wavenet](https://www.kaggle.com/nxrprime/wavenet-with-shifted-rfc-proba-and-cbr) as baseline. 
I found, that increasing receptive field before wave block improve CV and LB score (I think, this help because the length was too big).
So, I use maxpooling1d(2) with concat with signal before maxpool1d before wave block (max pool stride=1).

NN has 2 branch: wavenet and 2 bi-gru with concatenation, after that fully-connected layer.

I found, that Tversky loss with CCE improve NN accuracy and save training time.
Final loss was: CCE + focal + Tversky (without balancing).

I use GROUP BATCH SIZE=[4000, 50000, 25000, 12500, 10000, 8000, 20000] for prediction with with averaging.
