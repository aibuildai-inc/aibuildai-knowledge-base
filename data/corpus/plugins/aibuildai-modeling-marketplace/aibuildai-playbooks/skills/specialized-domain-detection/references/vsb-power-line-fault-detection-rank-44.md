# 44th solution... Thank you for your questions and advices

Competition: vsb-power-line-fault-detection
Rank: #44
Source: https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/85163#latest-496462

My public lb ranks 107th and private lb ranks 44th.
The train set of this competition is very small. And the distribution of train set is very different from the distribution of LB dataset. So I don't think local cv can display your models' performance on private LB. Maybe this competition's key is fitting public LB without overfitting.

My solution is as below:

- feature extraction
  My FE is similar as this [kenel](https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694/data). But there is some difference as below. 

  I exclude the trend of signal before feature extraction as below:
``` 
rollingdata = rawdata.rolling(20000, center=True, min_periods=1, axis=0)
trend = rollingdata.mean()  
res = (rawdata - trend).values
```

  In order to mine more information from raw signal, I divide raw signal into 250 segments.
```
    whole_signal_len = 800000
    processed_len = 250
    bin_len = int(whole_signal_len/processed_len)
```

  And I don't use 25% and 75% quantile. Because I think trend is useless for PD detection, and some quantiles are redund ant.
```
mean_slice = signal_slice.mean(axis=0)
std_slice = signal_slice.std(axis=0)

std_top = mean_slice + std_slice
std_bot = mean_slice - std_slice

percentil_slice = np.percentile(signal_slice, [0, 1, 50, 99, 100], axis=0)
max_range = percentil_slice[-1] - percentil_slice[0]

relative_percentile = percentil_slice - mean_slice


out_slice = np.concatenate([
                            percentil_slice.T,
                            max_range.reshape(-1, 1),
                            relative_percentile.T,
                            std_top.reshape(-1, 1),
                            std_bot.reshape(-1, 1),
                            std_slice.reshape(-1, 1),
                            mean_slice.reshape(-1, 1)
                            ], axis=1)
```

- model
There are two models in my solution.
```
class LSTM_softmax(nn.Module, myBaseModule):
    def __init__(self, random_seed,
                       features_dims,
                       seq_len,
                       learning_rate,
                       lstm_out_dim=80,
                       lstm_layers=2,
                       linearReduction_dim=64,
                       env=None):
        nn.Module.__init__(self)
        myBaseModule.__init__(self, random_seed)

        self.l2_weight = 0.0000
        self.lstm = nn.LSTM(features_dims, lstm_out_dim, lstm_layers, bidirectional=True, batch_first=True).cuda()
        self.lstm_attention = Attention(lstm_out_dim*2, seq_len)

        self.linear = nn.Linear(lstm_out_dim*4, linearReduction_dim).cuda()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.10)
        self.linear2 = nn.Linear(linearReduction_dim, 1).cuda()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch_num: 1/math.sqrt(epoch_num+1))
        self.loss_fn = torch.nn.BCELoss(reduction="sum")

        self.vis = env

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm_atten = self.lstm_attention(h_lstm)

        lstm_max_pool, _ = torch.max(h_lstm, 1)

        nn_out = torch.cat([lstm_max_pool, h_lstm_atten], 1)
        nn_out = self.dropout(nn_out)
        nn_out = self.relu(self.linear(nn_out))
        out = self.sigmoid(self.linear2(nn_out))

        return out

```
```
class LSTM_selfAttention_softmax(nn.Module, myBaseModule):
    def __init__(self, random_seed,
                       features_dims,
                       seq_len,
                       learning_rate,
                       lstm_out_dim=80,
                       lstm_layers=2,
                       selfAttention_dim=80,
                       linearReduction_dim=64,
                       env=None):
        nn.Module.__init__(self)
        myBaseModule.__init__(self, random_seed)

        self.l2_weight = 0.0000

        self.lstm = nn.LSTM(features_dims, lstm_out_dim, lstm_layers, bidirectional=True, batch_first=True).cuda()
        self.selfAttention = selfAttention(selfAttention_dim, selfAttention_dim, 2*lstm_out_dim,dk=64)

        self.lstm_attention = Attention(selfAttention_dim, seq_len)

        self.linear = nn.Linear(2*(selfAttention_dim), linearReduction_dim).cuda()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.10)
        self.linear2 = nn.Linear(linearReduction_dim, 1).cuda()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch_num: 1/math.sqrt(epoch_num+1))
        self.loss_fn = torch.nn.BCELoss(reduction="sum")

        self.vis = env

    def forward(self, x):

        h_lstm, _ = self.lstm(x)
        h_lstm = self.selfAttention(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        lstm_max_pool, _ = torch.max(h_lstm, 1)
        nn_out = torch.cat([lstm_max_pool, h_lstm_atten], 1)


        nn_out = self.dropout(nn_out)
        nn_out = self.relu(self.linear(nn_out))
        out = self.sigmoid(self.linear2(nn_out))
        return out
```
In order to reduce model complexity, I train 6 models for 3 phase data. Each phase data has 2 models. The final result is a weighted average of predictions of two models.



[github link](https://github.com/qq563902455/VSB_Power_Line_Fault_Detection)
If you think this is helpful, please star my github project and upvote my discussion.Thank you!!!
