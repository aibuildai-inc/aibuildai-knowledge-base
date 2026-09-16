# 8th place solution

Competition: trends-assessment-prediction
Rank: #8
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162799

We( @yasufuminakama, @berserker408 ) would like to thank kaggle &amp; host for the interesting competition and to all the participants for giving us a lot of ideas. And congrats to winners!

# Solution overview


# 3D map features
We extracted 512 3D map features using 3D-Resnet10. 
As I posted to External Data Thread, we used the one shared by @shentao ( see https://www.kaggle.com/c/trends-assessment-prediction/discussion/147797 ). More details, you can check his discussion. Thanks a lot to him.
As written in https://www.kaggle.com/c/trends-assessment-prediction/data, 
```
The scores (see train_scores.csv) are not the original age and raw assessment values. They have been transformed and de-identified to help protect subject identity and minimize the risk of unethical usage of the data. Nonetheless, they are directly derived from the original assessment values and, thus, associations with the provided features is equally likely.
Before transformation, the age in the training set is rounded to nearest year for privacy reasons. However, age is not rounded to year (higher precision) in the test set. Thus, heavily overfitting to the training set age will very likely have a negative impact on your submissions.
```
We need to avoid overfitting to the training data, to avoid overfitting to the training data, 3D map features helped a lot.

# Models
We used fillna data, fillna by SVM+Ridge.
## NN
NN base structure is below.
```
class CFG:
    hidden_size1=64
    hidden_size2=128
    hidden_size3=6
    dropout=0.5

class TabularNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
                          nn.Linear(26, CFG.hidden_size1),
                          nn.BatchNorm1d(CFG.hidden_size1),
                          nn.Dropout(CFG.dropout),
                          nn.PReLU(),
                          nn.Linear(CFG.hidden_size1, CFG.hidden_size1//2),
                          nn.BatchNorm1d(CFG.hidden_size1//2),
                          nn.Dropout(CFG.dropout),
                          )
        self.mlp2 = nn.Sequential(
                          nn.Linear(1378, CFG.hidden_size2),
                          nn.BatchNorm1d(CFG.hidden_size2),
                          nn.Dropout(CFG.dropout),
                          nn.PReLU(),
                          nn.Linear(CFG.hidden_size2, CFG.hidden_size2//2),
                          nn.BatchNorm1d(CFG.hidden_size2//2),
                          nn.Dropout(CFG.dropout),
                          )
        self.mlp3 = nn.Sequential(
                          nn.Linear(512, CFG.hidden_size3),
                          nn.BatchNorm1d(CFG.hidden_size3),
                          nn.Dropout(CFG.dropout),
                          nn.PReLU(),
                          nn.Linear(CFG.hidden_size3, CFG.hidden_size3//2+1),
                          nn.BatchNorm1d(CFG.hidden_size3//2+1),
                          nn.Dropout(CFG.dropout),
                          )
        self.head = nn.Sequential(
                          nn.Linear(CFG.hidden_size1//2+CFG.hidden_size2//2+CFG.hidden_size3//2+1, 
                                    CFG.hidden_size1//2+CFG.hidden_size2//2+CFG.hidden_size3//2+1),
                          nn.BatchNorm1d(CFG.hidden_size1//2+CFG.hidden_size2//2+CFG.hidden_size3//2+1),
                          nn.Dropout(0.1),
                          nn.Linear(CFG.hidden_size1//2+CFG.hidden_size2//2+CFG.hidden_size3//2+1, 5),
                          )
    def forward(self, cont_x):
        loading_x = cont_x[:, :26]
        fnc_x = cont_x[:, 26:1404]
        img_x = cont_x[:, 1404:]
        loading_x = self.mlp1(loading_x)
        fnc_x = self.mlp2(fnc_x)
        img_x = self.mlp3(img_x)
        x = torch.cat((loading_x, fnc_x, img_x), 1)
        x = self.head(x)
        return x
```
Criterion was optimized per target using optuna like below. This improved results.
```
def weighted_nae(inp, targ, W=torch.FloatTensor([0.3, 0.175, 0.175, 0.175, 0.175])):
    return torch.mean(torch.matmul(torch.abs(inp - targ), W / torch.mean(targ, axis=0)))

W_DICT = {'age': torch.FloatTensor([1.0, 0.28405462973430023, 0.15685327697616772, 
                                    0.06391563565928776, 0.014662813285223892]), 
          'domain1_var1': torch.FloatTensor([0.6638609260536665, 1.0, 0.14338389219696276, 
                                             0.1384448250515486, 0.1682642506073942]), 
          'domain1_var2': torch.FloatTensor([0.5528250413216551, 0.2767068350188637, 1.0, 
                                             0.7764813978084125, 0.1654236853816111]), 
          'domain2_var1': torch.FloatTensor([0.7856514193916039, 0.15941727005223633, 0.43970552609312813, 
                                             1.0, 0.16377864561256422]), 
          'domain2_var2': torch.FloatTensor([0.502415239502587, 0.2892349530842958, 0.2382098399995868, 
                                             0.15870234342595627, 1.0])}
```
Averaged NN scored CV: 0.15786, LB: 0.15781 without post-processing.
We prepared more averaged NN prediction and used it for final blend.

## other models
We referred four public kernels, and retrain them with our 3D map features.
https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging ( @aerdem4 )
https://www.kaggle.com/tunguz/rapids-ensemble-for-trends-neuroimaging ( @tunguz )
https://www.kaggle.com/david1013/trends-multi-layer-model ( @david1013 )
https://www.kaggle.com/andypenrose/baggingregressor-rapids-ensemble ( @andypenrose )
Thanks to these four kernels' authors.

## site2 detect model
This part is similar to the public kernel except adding the 3D map features. The valid AUC can reach 0.98. Thanks to Bojan.
https://www.kaggle.com/tunguz/adversarial-trends-site ( @tunguz )

# post-processing
This trick is quite common in regression competitions. And it improved our score by ~0.0003.
```
sub.loc[(sub.target=='age')&amp;(sub.site==1),'Predicted'] *= 1.01
sub.loc[(sub.target=='age')&amp;(sub.site==2),'Predicted'] *= 1.02
sub.loc[(sub.target=='domain1_var1'),'Predicted'] *= 1.01
sub.loc[(sub.target=='domain2_var1')&amp;(sub.site==2),'Predicted'] *= 1.01
```

# What did not work
- pseudo labeling

# Final result
Public score: 0.15683 (Public 10th) , Private score: 0.15729 (Private 8th)
