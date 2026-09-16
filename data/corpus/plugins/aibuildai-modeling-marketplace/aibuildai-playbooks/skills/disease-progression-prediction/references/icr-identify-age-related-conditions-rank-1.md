# How on Earth did I win this competetion?

Competition: icr-identify-age-related-conditions
Rank: #1
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/430843

Hello there! This was really unexpected. I hoped to be in top 10%, but never dreamed of something more. Thanks to everyone who participated in the competetion and especially to those who discussed various ideas and shared their code! And big thanks to SAMUEL, who introduced reweighting the probabilities in this notebook: https://www.kaggle.com/code/muelsamu/simple-tabpfn-approach-for-score-of-15-in-1-min

My solution was purely based on some sophisticated DNN: https://www.kaggle.com/room722/icr-adv-model
UPDATE: Training notebook attached.

What did not work for me:
1. Gradient boosting was obviously overfitting, although I spent just little time on it and didn't make much fine-tuning.
2. The "greeks" were useless. I think, because we have no greeks for the test data.
3. FE led to overfitting.

What did work:
1. DNN based on Variable Selection Network. [1]
2. No "casual" normalization of data like MinMaxScaler or StandartScaler, but instead a linear projection with 8 neurons for each feature.
3. Huge values of dropout: 0.75->0.5->0.25 for 3 main layers.
4. Reweighting the probabilities in the end worked really good.
5. 10 folds cv, repeat for each fold 10-30 times, select 2 best models for each fold based on cv (yes, cv somehow worked in this competition!).The training was so unstable, that the cv-scores could vary from 0.25 to 0.05 for single fold, partially due to large  dropout values, partially due to little amount of train data. That's why I picked 2 best models for each fold.
6. The cv was some kind of Multi-label. At first I trained some baseline DNN, gathered all validation data and labeled it as follows: (y_true = 1 and y_pred < 0.2) or (y_true = 0 and y_pred > 0.8) -> label 1, otherwise label 0. So, this label was somthing like "hardness to predict". And the other label was, of course, the target itself.

It will be honest to say, that I was lucky to win, but for me personally that means also that DNN wins, and probably not just by luck. And as a big fan of DNNs this makes me proud and happy))

[1] The idea of this network was taken from here https://arxiv.org/abs/1912.09363
