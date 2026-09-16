# 20th place solution

Competition: tlvmc-parkinsons-freezing-gait-prediction
Rank: #20
Source: https://www.kaggle.com/c/tlvmc-parkinsons-freezing-gait-prediction/discussion/416106

Thank you to the organizers for the fun competition and everyone who participated.
I share my solution.

# Summary
- dataset: tdcsofg + defog (not use notype/unlabeled)
- model: Conv1d NN (common model for tdcsfog and defog) 
- 6 models ensemble

# 1. baseline model
- I use this notebook. [PyTorch FOG End-to-End Baseline [LB 0.254] ](https://www.kaggle.com/code/mayukh18/pytorch-fog-end-to-end-baseline-lb-0-254)
- Thank you for @mayukh18 

# 2. Validation Strategy
- GroupKFold : 5-fold (groups=Subject)
- Since it was imbalanced data, I adjusted the seed so that the number of cases and the ratio would be the same as much as possible

# 3. Preprocess data
- Align the units of acceleration (divided by 9.8066 for tdcsfog). 
- features:
  - raw data: AccV, AccML, AccAP
  - AccMG: np.sqrt(AccV^2 + AccML^2 + AccAP^2)
  - Time_freq: df["Time"] / df["Time"].max()
  - tdcs flag
  - Rolling Window Features

```
# Rolling Window Features
for col in ["AccV", "AccML", "AccAP", "AccMG"]:
    for w in [10, 50, 100, 1000]:
        df[f"{col}_win{w}_std"] = df[col].rolling(window=w, min_periods=1).std().fillna(0)
        df[f"{col}_win{w}_delta"] = df[col].rolling(window=w, min_periods=1).max() - \
                                    df[col].rolling(window=w, min_periods=1).min()
        df[f"{col}_win{w}_diff"] = df[col] - df[col].rolling(window=w, min_periods=1).mean()
```

# 4. Model
- I define 2 models.
  - model A: 
      - input's feature: 54 features (all feature)
      - window_size=32: past=24, future=8, wx=8
      - Input - (Conv1d:ks=3/5/10 - GAP) x 3 - MLP - Outputs
  - model B:
      - input's feature: 6 features (3raw + MG + Time_freq + tdcs)
      - window_size=256: past=192, future=64, wx=1
      - Input - Conv1d:ks=90 - GAP - MLP - Outputs
- training parameter:
  - loss: BCEWithLogitsLoss / BCE+CELoss(4label)
  - optimizer: Adam(2e-5)
  - epoch: 10

[model.jpg]

# 5. Postprocess
- moving average of predicted values: window=500
- label 1 is continuous so I could increase score just a little. (+0.004)

# 6. Ensemble
- I trained 6 models.
- The ensemble method is an equally weighted average.

| id | inputs| win_size | Acc-units | model | local-cv | public | private |
|---|---|---:|---|---:|---:|---:|
| v10 | 54feats |32(wx=8) | None | modelA | 0.290 | 0.392 | 0.316 |
| v13 | 54feats |32(wx=8) | None | modelA (custom) | 0.300 | 0.376 | 0.305 |
| v51b | 54feats + 2feats |48(wx=8) | div by 9.8| modelA | 0.302 | 0.386 | 0.321 |
| v60 | 54feats |32(wx=8)| div by 9.8|  modelA (+CELoss) | 0.301 | 0.388 | 0.306 |
| v59 | 54feats |32(wx=8) |None | modelA (other cv) | 0.324 | 0.390 | 0.312 |
| v58 | 3raw+3feats |256(wx=1) | div by 9.8| modelB | 0.306 | 0.350 | 0.312 |
|  |  |  |   | | ensemble  | 0.411 | 0.324 |

# Did't work
- Transformer / LSTM / 1dcnn+LSTM / wavenet :  did not improve score
- Data conversion by Fourier transform / MFCC:  score got worse
- RobustScaler: almost the same
- pseudo-labeling for notype and unlabeled-data: score got worse
- use Subject data: score got worse
- separate model for tdcsfog and defog: score got worse

There were a few methods that worked for the top teams. I may have done something wrong.

# My question
- Why did I get similar scores with and without matching acceleration units?  (Sometimes it is better not to match)
- I didn't want to include Time_freq to make it a generic model. How can I improve my score without this?

Thank you for reading.
