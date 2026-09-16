# 12th solution: single model wavenet + lstm

Competition: liverpool-ion-switching
Rank: #12
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153993

Firstly thanks a lot to my teammates and the great community, to all people who keep sharing, we have all learned a lot during this competition. (I haven't even heard wavenet before this, neither have I imagined to solve time-series problem using CNN)

Yesterday at 2am the private LB was revealed and after that I was too excited. I just slept for 2 hours this morning even though today I have to work, that's for our one month's hard work and finally I got my first gold medal.

## Some thoughts
- We were basically focusing on doing extra work on the famous 0.944 wavenet RFC kernel.
- We found groupKFold CV of group size 4000 was consistent to the public LB, but not perfectly consistent. Then we found that the class 10 count in test set was very informative, we only submit when test class 10 count is over 7000, ideally about 7020. By doing this our CV strategy became quite consistent, this allowed us to do all kinds of experiments in local securely. Different batch size or group size didn't help.
- Our final two submissions are this one (highest CV score) and a vote of three submissions (highest LB score), this submission is finally also the highest on private LB, while the vote dropped violently. This in return supports the fact that our CV strategy worked well.
- Flipping helped, but adding gaussian noise didn't, we didn't try augmentation at test time.
- We didn't spend time on HMM modeling. 
- Our best LGB model was 0.941 on public LB, as someone mentions in a post that RFC leaks because of the different CV split strategy, we tried to replace it by LGB/RFC/LSTM using the same groupKFold, but no luck.

## Preprocessing
- We used drift cleaned and kalman filtered signal
- We dropped 200,000 rows from 3,640,000 to 3,840,000 , as to remove the spike noises. That made no difference on public LB score.

## Features
We tried a lot features but few of them work, then we decided to stick with model structure tuning, hoping it to find useful features by itself.
Still, except the RFC probabilities, some of features worked, even not bringing big boost.
So we used these features in this model:
- RFC Probabilities
- signal ** 2
- lag range 3 and -11, -15: [-15, -11, -3, -2, -1, 1, 2, 3]
All features are created after standard scaling signal.

## Model structure (Pytorch)
`class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate, padding_mode='replicate'))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate, padding_mode='replicate'))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res
    
class SEModule(nn.Module):

    def __init__(self, in_channels, reduction=2):
        super(SEModule, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        s = F.adaptive_avg_pool1d(x, 1)
        s = self.conv(s)
        x *= torch.sigmoid(s)
        return x

class Classifier(nn.Module):
    
    def __init__(self, inch=8, kernel_size=3):
        super().__init__()
        dropout_rate = 0.1
        
        self.conv1d_1 = nn.Conv1d(inch, 32, kernel_size=1, stride=1, dilation=1, padding=0, padding_mode='replicate')
        self.batch_norm_conv_1 = nn.BatchNorm1d(32)
        self.dropout_conv_1 = nn.Dropout(dropout_rate)
        
        self.conv1d_2 = nn.Conv1d(inch+16+32+64+128, 32, kernel_size=1, stride=1, dilation=1, padding=0, padding_mode='replicate')
        self.batch_norm_conv_2 = nn.BatchNorm1d(32)
        self.dropout_conv_2 = nn.Dropout(dropout_rate)
        
        self.wave_block1 = Wave_Block(32, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(inch+16, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(inch+16+32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(inch+16+32+64, 128, 1, kernel_size)
        
        self.se_module1 = SEModule(16)
        self.se_module2 = SEModule(32)
        self.se_module3 = SEModule(64)
        self.se_module4 = SEModule(128)        
        
        self.batch_norm_1 = nn.BatchNorm1d(16)
        self.batch_norm_2 = nn.BatchNorm1d(32)
        self.batch_norm_3 = nn.BatchNorm1d(64)
        self.batch_norm_4 = nn.BatchNorm1d(128)
        
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)
        self.dropout_4 = nn.Dropout(dropout_rate)
        
        self.lstm = nn.LSTM(inch+16+32+64, 32, 1, batch_first=True, bidirectional=True)
        self.fc0 = nn.Linear(64, 32)
        self.fc1 = nn.Linear(32, 11)
        self.fc2 = nn.Linear(32, 11)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x0 = self.conv1d_1(x)
        x0 = F.relu(x0)
        x0 = self.batch_norm_conv_1(x0)
        x0 = self.dropout_conv_1(x0)
        
        x1 = self.wave_block1(x0)
        x1 = self.batch_norm_1(x1)
        x1 = self.dropout_1(x1)
        x1 = self.se_module1(x1)
        x2_base = torch.cat([x1, x], dim=1)
        
        x2 = self.wave_block2(x2_base)
        x2 = self.batch_norm_2(x2)
        x2 = self.dropout_2(x2)
        x2 = self.se_module2(x2)
        x3_base = torch.cat([x2_base, x2], dim=1)
        
        x3 = self.wave_block3(x3_base)
        x3 = self.batch_norm_3(x3)
        x3 = self.dropout_3(x3)
        x3 = self.se_module3(x3)
        x4_base = torch.cat([x3_base, x3], dim=1)
        
        x4 = self.wave_block4(x4_base)
        x4 = self.batch_norm_4(x4)
        x4 = self.dropout_4(x4)
        x4 = self.se_module4(x4)
        x5_base = torch.cat([x4_base, x4], dim=1)
        
        x5 = self.conv1d_2(x5_base)
        x5 = F.relu(x5)
        x5 = self.batch_norm_conv_2(x5)
        x5 = self.dropout_conv_2(x5)

        lstm_out, _ = self.lstm(x4_base.permute(0, 2, 1))
        out1 = self.fc0(lstm_out)
        out1 = self.fc1(out1)
        out2 = x5.permute(0, 2, 1)
        out2 = self.fc2(out2)
        return out1 * out2 `

- Wave block is the same in public kernel. The concatenation after se module in each block gave us a boost, so does the lstm layer.
- we've also tried multi-task learning, using open_channels both as classes and as continuous values, but no luck with this model. Interestingly we have another LSTM model without wavenet, using this on that model gave us a huge boost of 0.002 on public LB.
- tried 1d pooling encoder decoder, didn't help

## About model training
We were using CrossEntropyLoss as loss function, we did try other functions like focal loss and ohem loss, but no luck.
Learning rate is 0.001, tried weight_decay but didn't help, batch size is 16.
This part I believe helped a lot, as CE loss is not quite consistent with macro f1 score :
- save model weight each time macro F1 score improves
- after 10 epochs that macro F1 doesn't improve, and CE loss doesn't reduce, learning rate /= 2, and load lastly saved model (best macro F1 so far)
- after 20 epochs that macro F1 doesn't improve, and CE loss doesn't reduce, early stopping
- either macro f1 improves or CE loss reduces, epoch counter = 0
We believe this will give the model more chances to optimize F1 score, and we retrain a fold if its score is relatively lower than before, until it can give us satisfying score (usually 1 to 2 times).
Then the training procedure will become like this: (red means F1 score improves, green means CE loss)



Finally this model got 0.94365 on local CV, 0.94545 on public LB and 0.94513 on private LB.


**Edit 1:**
I've seen that some people are interested in the LSTM model, so I can show you this simple multi-task learning model, even though there's no magic in it.


`class Classifier(nn.Module):

    def __init__(self, inch=8):
        super().__init__()        
        self.conv_7 = nn.Conv1d(inch, 12, kernel_size=7, stride=1, padding=3, padding_mode='replicate')
        self.conv_5 = nn.Conv1d(inch, 12, kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        self.conv_3 = nn.Conv1d(inch, 12, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        
        self.lstm = nn.LSTM(48, 45, 2, batch_first=True, bidirectional=True)

        self.linear_1 = nn.Linear(90, 45)
        self.linear_2 = nn.Linear(45, 22)
        
        self.out_cls = nn.Linear(22, 11)
        self.out_reg = nn.Linear(22, 1)
        
        for l in [self.linear_1, self.linear_2, self.out_cls, self.out_reg]:
            nn.init.kaiming_normal_(l.weight.data)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_out_7 = self.conv_7(x)
        conv_out_5 = self.conv_5(x)
        conv_out_3 = self.conv_3(x)
        
        concated_out = torch.cat([x, conv_out_3, conv_out_5, conv_out_7], 1)
        
        lstm_out, _ = self.lstm(concated_out.permute(0, 2, 1))

        out_tmp_1 = F.relu(self.linear_1(lstm_out))
        out_tmp_2 = F.relu(self.linear_2(out_tmp_1))
        y_pred_cls = self.out_cls(out_tmp_2)
        y_pred_reg = self.out_reg(out_tmp_2)
        return y_pred_cls, y_pred_reg`

I use CrossEntropyLoss for the classification output and MSELoss for the regression output.
This model can achieve 0.94487 on the public LB, but only 0.94388 on the private LB
