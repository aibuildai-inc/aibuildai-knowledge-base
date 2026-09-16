# 2nd place solution

Competition: nfl-big-data-bowl-2026-prediction
Rank: #2
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2026-prediction/writeups/2nd-place-solution

Many thanks the competition hosts and Kaggle for organizing this exciting challenge. 

##Solution Overview
#Keypoints

I learned some useful settings from this notebook: https://www.kaggle.com/code/ryanadamsai/nfl-big-data-bowl-geometric-gnn-lb-586

- using raw input as the input of model
    - I have no features, the features from public notebooks are too complicated for me.
        - The major inputs are 's', 'a',  'dir', 'o',  'x',  'y',  and other static information
    - Finally I used only a small subset of simple features from this notebook, and I did not find complicate features helpful.
    - Since the calculation of features are very slow, I used raw input to accelerate the pipeline.

- predicting the difference (delta x, delta y) from the previous frame
    - it is interesting however very helpful
    - predicting the direction and velocity yields worse result

- loss function copied from high score public notebook with muon optimizer
    - since it helped to improve my results, I kept it until the end of the competition
```
class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
    
    def forward(self, pred, target, mask):
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(abs_err <= self.delta, 0.5 * err * err, 
                           self.delta * (abs_err - 0.5 * self.delta))
        
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1) * weight
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)
```
- horizontal/vertical flip augmentation helped a lot
- simple normalization like BN/dropout helps
    - advanced method like rdrop does not help
- ensembling models helped slightly, but always helped. So I trained more models the last few days. 
    - everytime I add a model, the score improves.

# Training
- train with 2021/ 2023 data added
    - pretrain does not help for me, just train it.
- EMA to make the training stable
    - otherwise the validation score is very volatile.
- 5 last week as the oof
    - did not use kfold for training for faster iterations

## Model Architecture
My model design is almost the same as the [5th place solution](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/5th-place-solution)
- raw input
- few feature engineerings using pytorch operations
- RNN encoder
    - transformer encoder does not work
    - adding cnn1d does not help for me
- spatial temperal encoder
- decoder head which predicts the difference of movement, then cumsum as the prediction


## Other things
I noticed that the host was very helpful and active. However, he talked about improving the way they would use to sample the private test dataset, which made the final result unstable since "the distribution" would change. It made the test dataset quality better however harder to predict the future performance of models.

**ps: I made the post very late and went to sleep. Then as was kindly requested by some kagglers, I have enriched some details.**
