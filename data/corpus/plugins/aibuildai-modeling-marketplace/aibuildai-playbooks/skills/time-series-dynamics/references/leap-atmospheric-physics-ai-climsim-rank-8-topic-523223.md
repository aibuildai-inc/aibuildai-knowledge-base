# 8th solution

Competition: leap-atmospheric-physics-ai-climsim
Rank: #8
Source: https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523223

First of all, I would like to thank the organizers and Kaggle for hosting this competition. The quality of the competition data is great. Although there were some problems during the process, in any case, we have achieved results that satisfy most people.

This is my first solo gold medal and I became the Competition GrandMaster. This seven-year journey has been quite long and exciting.

### solution summary 

I think my solution is very simple, and it is basically based on the seq2seq models derived from BiLSTM.

(bs, 60, 25)  --> seq2seq --> (bs, 60, 14) --> (bs, 368) 



### models

* validate on last 6 months sample data

| models | CV |  LB |
| --- | --- | --- |
| BiLSTM(layers=6) | 0.7844 | 0.7812 |
| BiGRU(layers=8) | 0.7835 | 0.7802 |
| BiLSTM+Transformer | 0.7858 | 0.7821 |
| BiLSTM+Attention | 0.7865 | 0.7834 |
| BiLSTM+TCN | 0.7855 | 0.7832 |
| BiLSTM+CNN | 0.7842 | 0.7821 |
| ensemble on models | 0.7923 | 0.7890 |
| ensemble on targets | 0.7933 | 0.7884 |


- A base BiLSTM model:

```
class LeapModel(nn.Module):
    def __init__(self,
                 input_size,
                 seq_len,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=.3,
                 hidden_layers=[128, 256]):

        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional=bidirectional
        self.output_size=output_size

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=dropout)

        if hidden_layers and len(hidden_layers):
            first_layer  = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_layers[0])
            self.hidden_layers = nn.ModuleList(
                [first_layer] + \
                [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers:
                nn.init.kaiming_normal_(layer.weight.data)
            self.intermediate_layer = nn.Linear(hidden_layers[-1], self.input_size)
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)
        else:
            self.hidden_layers = []
            self.intermediate_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_size, self.input_size)
            self.output_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.activation_fn = torch.nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs, hidden = self.rnn(x)

        x = self.dropout(self.activation_fn(outputs))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)

        # (-1,60,14) -> (-1,386)
        o_s = x[:, :, :6]
        o_s = o_s.permute(0,2,1).reshape(-1,360)
        o_g = x[:, :, 6:]
        o_g = o_g.mean(dim=1)
        out = torch.cat([o_s, o_g], dim=1)

        return out

input_size = 25
output_size = 14
seq_len = 60

hidden_size = 256
hidden_layers = [256, 512]
num_layers = 6
dropout = 0.1

model = LeapModel(
    input_size=input_size,
    seq_len=seq_len,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    hidden_layers=hidden_layers,
    dropout=dropout,
    bidirectional=True,
).to(device)
```

Reference Links: https://www.kaggle.com/code/brandenkmurray/seq2seq-rnn-with-gru

- A BiLSTM with Transformer model

```
class LeapModel(nn.Module):
    def __init__(self,
                 input_size,
                 seq_len,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.3,
                 hidden_layers=[128, 256],
                 nhead=8,
                 num_transformer_layers=2):

        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size

        # LSTM layer
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=dropout)

        # Transformer layer
        transformer_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_input_size, nhead=nhead, dropout=dropout),
            num_layers=num_transformer_layers
        )

        # Fully connected layers
        if hidden_layers and len(hidden_layers):
            first_layer = nn.Linear(transformer_input_size, hidden_layers[0])
            self.hidden_layers = nn.ModuleList(
                [first_layer] + \
                [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers:
                nn.init.kaiming_normal_(layer.weight.data)
            self.intermediate_layer = nn.Linear(hidden_layers[-1], self.input_size)
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)
        else:
            self.hidden_layers = []
            self.intermediate_layer = nn.Linear(transformer_input_size, self.input_size)
            self.output_layer = nn.Linear(transformer_input_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.activation_fn = torch.nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM layer
        lstm_output, _ = self.rnn(x)
        
        # Transformer layer
        transformer_output = self.transformer_layer(lstm_output)
        
        # Apply dropout and activation
        x = self.dropout(self.activation_fn(transformer_output))
        
        # Fully connected layers
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)

        # Reshape output
        o_s = x[:, :, :6]
        o_s = o_s.permute(0, 2, 1).reshape(-1, 360)
        o_g = x[:, :, 6:]
        o_g = o_g.mean(dim=1)
        out = torch.cat([o_s, o_g], dim=1)

        return out

input_size = 25
output_size = 14
seq_len = 60

hidden_size = 256
hidden_layers = [256, 512]
num_layers = 6
dropout = 0.1
nhead = 8
num_transformer_layers = 1

model = LeapModel(
    input_size=input_size,
    seq_len=seq_len,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    hidden_layers=hidden_layers,
    dropout=dropout,
    bidirectional=True,
    nhead=nhead,
    num_transformer_layers=num_transformer_layers
).to(device)
```

- A BiLSTM with TCN model

```
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=(kernel_size-1) * dilation // 2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation_fn = nn.GELU()

    def forward(self, x):
        return self.activation_fn(self.bn(self.conv(x)))


class LeapModel(nn.Module):
    def __init__(self,
                 input_size,
                 seq_len,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.3):

        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size

        # LSTM layer
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=dropout)

        self.se = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size//2),
            nn.GELU(),
            nn.Linear(hidden_size//2, hidden_size*2),
            nn.Sigmoid()
        )
        
        self.tcn = nn.Sequential(
            TCNBlock(hidden_size*2, hidden_size*2, kernel_size=3, dilation=1),
            TCNBlock(hidden_size*2, hidden_size*2, kernel_size=3, dilation=2),
            TCNBlock(hidden_size*2, hidden_size*2, kernel_size=3, dilation=4),
            TCNBlock(hidden_size*2, hidden_size*2, kernel_size=3, dilation=8),
        )
        
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        # RNN layer
        outputs, _ = self.rnn(x)
        
        se_weights = self.se(torch.mean(outputs, dim=1)).unsqueeze(1)
        outputs = outputs * se_weights
        
        tcn_input = outputs.permute(0, 2, 1)
        tcn_output = self.tcn(tcn_input)
        tcn_output = tcn_output.permute(0, 2, 1)
        
        x = self.dropout(tcn_output)
        x = self.fc(x)

        # Reshape output
        o_s = x[:, :, :6]
        o_s = o_s.permute(0, 2, 1).reshape(-1, 360)
        o_g = x[:, :, 6:]
        o_g = o_g.mean(dim=1)
        out = torch.cat([o_s, o_g], dim=1)  # (bs,368)

        return out


input_size = 25
output_size = 14
seq_len = 60

hidden_size = 256
num_layers = 6
dropout = 0.1

model = LeapModel(
    input_size=input_size,
    seq_len=seq_len,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout=dropout,
    bidirectional=True,
).to(device)
```

- A BiLSTM with Attention model

```
class LeapModel(nn.Module):
    def __init__(self,
                 input_size,
                 seq_len,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=.3,
                 hidden_layers=[128, 256]):

        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional=bidirectional
        self.output_size=output_size
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=0.1)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2 if bidirectional else hidden_size,
                                               num_heads=8,
                                               batch_first=True)

        if hidden_layers and len(hidden_layers):
            first_layer  = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_layers[0])
            self.hidden_layers = nn.ModuleList(
                [first_layer] + \
                [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers:
                nn.init.kaiming_normal_(layer.weight.data)
            self.intermediate_layer = nn.Linear(hidden_layers[-1], self.input_size)
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)
        else:
            self.hidden_layers = []
            self.intermediate_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_siz, self.input_size)
            self.output_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.activation_fn = torch.nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        outputs, hidden = self.rnn(x)

        outputs = outputs.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        attn_output, _ = self.attention(outputs, outputs, outputs)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)

        x = self.dropout(self.activation_fn(attn_output))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)

        # (-1,60,14) -> (-1,386)
        o_s = x[:, :, :6]
        o_s = o_s.permute(0,2,1).reshape(-1,360)
        o_g = x[:, :, 6:]
        o_g = o_g.mean(dim=1)
        out = torch.cat([o_s, o_g], dim=1)

        return out


input_size = 25
output_size = 14
seq_len = 60

hidden_size = 256
hidden_layers = [256, 512]
num_layers = 6
dropout = 0.1

model = LeapModel(
    input_size=input_size,
    seq_len=seq_len,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    hidden_layers=hidden_layers,
    dropout=dropout,
    bidirectional=True,
).to(device)
```

### Dataset

1. Download all 0001-02 to 0009-01  (low resolution) data from Huggingface.
2. 0001-02 to 0008-06 as training set, 0008-07 to 0009-01 as validate set (sampling to ~625000 rows).

### Some training details

1. Loss: nn.SmoothL1Loss(reduction='mean') (0.005~0.008 better than mse)
2. Scheduler: get_cosine_schedule_with_warmup
3. Activation Function: GELU (0.002~0.004 better than relu)
4. Trained on 4*RTX4090 with 360G RAM, 7.5 years training dataset, ~1 hour per epoch



### Post-Processing

```
targets_unpredictable = []
for target in weights:
    if weights[target] == 0.:
        targets_unpredictable.append(target)
for target in targets_unpredictable:
    df_pred[target] = 0.
for target in [f'ptend_q0002_{i}' for i in range(12, 28)]:
    df_pred[target] = -df_test[target.replace("ptend", "state")] * weights[target] / 1200.
```

Reference Links: https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484

### Ensemble

1. on models: w0 * pred0 + w1 * pred1 + ... + w5 * pred5
2. on targets:

```
selects = []
for idx_t, target in tqdm(enumerate(TARGETCOLS), total=len(TARGETCOLS)):
    di = {}
    for idx_p, prob in enumerate(probs):
        di[idx_p] = r2_score(df_valid[target], probs[idx_p][:, idx_t])
    selects.append(sorted(di, key=di.get, reverse=True)[:4])
````

### What didn't work

1. All 8 years dataset with fix epochs, without validation.
2. Data augment: mask 10% input and TTA.
