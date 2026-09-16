# 17th place solution - 4 features model

Competition: riiid-test-answer-prediction
Rank: #17
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209713

My solution is based on a stack of the 3 models:

Edit: [Code](https://github.com/NikolaBacic/riiid)

Transformer(Encoder) - Validation: 0.8100
LSTM - Validation: 0.8062
GRU - Validation: 0.8060

LightGBM for stack: Validation: 0.8119 LB: 0.814x

I validated on new users (2.5M rows). [This one](https://www.kaggle.com/its7171/cv-strategy) is great, but it was computationally expensive for me. 

All three models used 4 features (embeddings):
- question id embedding
- response of the previous question (1-correct 0-incorrect)
- **ln(lag+1) * minute_embedding : taking a ln(x+1) of the lag initialy improved my score by ~0.01. I'd be very interested to hear whether it'd improve your models too**
- ln(prior_elapsed_time+1) * minute_embedding

Input is sum of these 4 embeddings.

**Parameters**

Shareable parameters (all three models):
max_quest = 300 (window size)
slide = 150
Adam optimizer
cosine lr scheduler
BCE loss
**xavier_uniform_ weight initialization (0.004 improvement over PyTorch's default one)**

I used Optuna for hyperparameter search on 20% of the data. It was my first time using it, and it's a great tool!

Transformer hyperparameters:
nhead = 8
head_dim = 60
dim_feedforward = 2048
num_encoder_layers = 8
epochs = 6
batch_size = 64
lr = 0.00019809259513409007
warmup_steps = 150*5

LSTM hyperparameters:
input_size_lstm = 384
hidden_size_lstm = 768
num_layers_lstm = 4
epochs = 4
batch_size = 64
lr = 0.0007019926812886481
warmup_steps = 100

GRU hyperparameters:
input_size_gru = 320
hidden_size_gru = 512
num_layers_gru = 3
epochs = 4
batch_size = 64
lr = 0.0008419253431185227
warmup_steps = 80

What didn't work:
-lectures
-questions metadata
-position encoding: both learnable and hard-coded
-predicting user_answer instead of correctness
-gradient clipping
-label smoothing
-dropout
-...

gg
