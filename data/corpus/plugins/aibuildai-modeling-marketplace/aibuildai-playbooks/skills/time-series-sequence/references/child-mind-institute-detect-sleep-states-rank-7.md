# 7th Place Solution - Wavenet and Some Tricks

Competition: child-mind-institute-detect-sleep-states
Rank: #7
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459598

First of all, thanks my teammate @robikscube . It was a fun ride. We have noticed that Public LB is not very trustworthy and focused on our CV. We had around 0.826 CV at the end.

**Our main model is a wavenet with some modifications.** It works on a sequence of minutely aggregated 3 days.

```python
class SleepModel(nn.Module):
    def __init__(self, inch, kernel_size):
        super(SleepModel, self).__init__()
        emb_size = 4
        
        self.minute_emb = nn.Embedding(15, embedding_dim=emb_size)
        
        self.bn = nn.BatchNorm1d(inch-1)
        self.wave_block1 = Wave_Block(inch-1+emb_size, 32, 8, kernel_size)
        self.wave_block2 = Wave_Block(32, 32, 7, kernel_size, base=2.25)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=5, dilation=DAY_LEN//6), 
                                         nn.BatchNorm1d(64), nn.LeakyReLU(),
                                         nn.Conv1d(64, 64, kernel_size=5, dilation=DAY_LEN//3), 
                                         nn.BatchNorm1d(64), nn.LeakyReLU()
                                        )
        self.top = nn.Conv1d(64, 3, kernel_size=1)
        self.top2 = nn.Conv1d(64, 2, kernel_size=1)
        
        
        self.gn1 = nn.GroupNorm(4, 32)
        self.gn2 = nn.GroupNorm(4, 32)
        
        self.avgpool = nn.AvgPool1d(kernel_size=DAY_LEN+1, padding=DAY_LEN//2, stride=1)
        self.maxpool = nn.MaxPool1d(kernel_size=DAY_LEN+1, padding=DAY_LEN//2, stride=1)
        

    def forward(self, x):
        x = torch.cat([x[:, -DAY_LEN:]*0, x, x[:, :DAY_LEN]*0], axis=1)
        x = x.permute(0, 2, 1)
        minute, x = x[:, -1, :], x[:, :-1, :]
        
        x[:, -1, :] += minute / 60 # hour + minute/60
        x = self.bn(x)
        
        minute_emb = self.minute_emb(torch.fmod(minute, 15).long()) # minute % 15 is an important feature
        x = torch.cat([x, minute_emb.permute(0, 2, 1)], axis=1)

        x = self.wave_block1(x)
        x = self.gn1(x)
        x = self.wave_block2(x)
        x = self.gn2(x)
        x = self.wave_block3(x)
                
        x = torch.cat([x, self.avgpool(x[:, :32]), self.maxpool(x[:, 32:])], axis=1)

        x = self.wave_block4(x)
        x, x2 = self.top(x), self.top2(x)
        return x, x2
```

Its **features** are: ```
["target", "idx", "anglez_mean", "anglez_std", 
                                                       "enmo_mean", "enmo_std", 
                                                       "same_anglez_prev_min", "same_anglez_next_min"] +
                                                       volatility_cols + ["hour", "minute"]
```
Volatility columns are median over anglez absolute difference on 3 different time windows. (5, 30, 480 steps)
Same anglez prev and next features are difference of anglez with the last day same minute.

**Training details:**
* Target is also set to 1 for the adjacent minute because we predict with +3 step offset to cover the next minute too.
* Ignoring near misses in loss: 2nd and 3rd minutes around the target are set to -1. SO that we dont penalize positive predictions there.
* 2 heads (1 for actual target -> categorical crossentropy, 1 for 15min windows -> BCE)
* Online Hard Example Mining with 50%
* 6 epochs with diminishing LR. Each epoch takes 18 seconds on RTX3090. (and submission was taking 30 minutes in total)
* 2 iterations with different seed for diversity. 1st iteration excludes some bad series and 2nd iteration has very little augmentation.

**Prediction details:**
* 1 day sliding predictions with 3 days span. Only the middle day is used.
* Postprocessing: Starting from the max prediction, we add their location as prediction. Score is their probability and 2nd highest probability within +-4 minute window. We set these windows to zero. We multiply surrounding +-20 minutes windows by 0.5 and continue. Later, we sample very low probability predictions for +-18 steps span for predictions with high score.
* Stacking with LightGBM: Lightgm with a lot of median absolute anglez difference features are used. NN probabilities and time to/till onset/wakeup features using this probabilities are added. AUC got significantly improved but the metric is improved very little. (0.001)
* Ensembling: We had another model with LSTM+Transformer. We added it to get 0.001 improvement.
