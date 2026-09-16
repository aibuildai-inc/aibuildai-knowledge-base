# 5th Place Solution - Team 💳VISA💳(Patrick's part)

Competition: amex-default-prediction
Rank: #5
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/348118

URL to the Summary&zakopuro part: https://www.kaggle.com/competitions/amex-default-prediction/discussion/348097

To begin with, I would like to thank Amex for hosting the competition and my teammates ( @zakopur0 @scumufeng @baosenguo). Congrats to @zakopur0 and @scumufeng for being a Kaggle competition master!

# **Pretrain + Finetune approach**
LightGBM + Feature engineering is very successful in this competition, and they outperform NN + raw features most of the time. Because of this, I believe the features used in LGBM models are very powerful, so I decided to first let the NN learn how to do feature engineering in the pretrain stage, then finetune the model with the target after that.
In this way, we provide much more guidance to train the model by using thousands of features, and most importantly we can include test data in the pretrain stage.
With pretraining, the model can have **+0.002** boost in public LB compared with training from scratch, and we can train a larger model (6 layers transformer) without any problem.




# **Model architecture**
We use different MLP layers to handle different types of inputs (Delinquency, Spend, Payment, Balance, Risk variables), then concatenate them and pass them to the transformer encoder. After the encoding part, we take the latest node and get the outputs through a linear layer.

# **Pretrain stage**
In the pretrain stage, our target is tabular features. We use Huber loss to train the standardized target and won't pass any loss if the feature is nan. The number of epochs is around 200 in this stage.

# **Finetune stage**
In the finetune stage, we train the model with the label. With pretraining, the converging speed for the model is very fast and we only need less than 5 epochs per fold in this stage!

# **Model performance**
Our best Transformer model without meta features has **0.794** CV, **0.796** public LB, and **0.804** private LB.
With pseudo labeling (We use soft predictions from our best ensemble model), a single transformer model can have **0.800** public LB and **0.808** private LB. (There is leaking because we didn't use Nested K-fold CV to generate pseudo label).
