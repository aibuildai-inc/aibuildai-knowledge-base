# 15th place summary

Competition: birdclef-2022
Rank: #15
Source: https://www.kaggle.com/c/birdclef-2022/discussion/327393

Thanks to team member and all participants.
I enjoyed the speech recognition competition very much,
but I couldn’t grasp the behavior of the metric and LB until the end.
Also, crinoid ’s EDA was very helpful in formulating a strategy.
The detailed solution is shown below:

**Melspectrogram preprocessing parameter**
```
sr = 32000
n_mels = 128
fmin = 20
fmax = 16000
n_fft = 32000//10
hop_len = 3200//4
power = 2
top_db = 80
```
**Model Architecture**
we was using very simple CNN architecture.
```
class Model(nn.Module):
    def __init__(self,name,pretrained=False):
        super(Model, self).__init__()
        self.model = timm.create_model(name,pretrained=pretrained, in_chans=1)
        self.model.reset_classifier(num_classes=0) 
        in_features = self.model.num_features
        self.fc = nn.Linear(in_features, cfg.CLASS_NUM)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
```

As a backbones we have used:
・eca_nfnet_l0
・convnext_tiny
・resnest50d
Model Training
we have trained models on 5 sec and 7sec chunks and with secondary labels.
At First, we made pretrain models by 2020, 2021, 2022 without scored birds competition data.
Next, we have finetunned pretrain models the scored birds datas.

**Ensemble and TTA**
We ensemble the single trained model (head 30s train tail5s validation 7s segment) and 5foldmodel (5s segment) and apply TTA:
```
if i <= 2:
                pred = model(images).sigmoid().detach().cpu().numpy()
            else:
                pred1 = model(images[:,:,:,0:201]).sigmoid().detach().cpu().numpy()
                pred2 = model(images[:,:,:,40:241]).sigmoid().detach().cpu().numpy()
                pred3 = model(images[:,:,:,80:281]).sigmoid().detach().cpu().numpy()
                pred = 0.25*pred1 + 0.5*pred2 + 0.25*pred3
```

**Post Processing**

Model weighted
Models with high LB are weighted larger and models with lower LB are weighted smaller.

and then we apply voting method:
After weight averaging the model’s predictions, we conducted pred re-weighting by voting score.  Voting score was calculated by judging ,for each species, whether each model’s pred exceeds the given threshold or not. By doing this, we aimed to grab more rare bird’s TP.

how to decide threshold
we apply percentaile approach(2021 2nd).
