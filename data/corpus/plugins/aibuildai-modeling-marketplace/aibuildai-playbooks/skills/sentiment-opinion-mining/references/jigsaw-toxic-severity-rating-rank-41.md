# 41st solution

Competition: jigsaw-toxic-severity-rating
Rank: #41
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306171

Congradulations, all winners.  
I learned a lot about NLP in this competition. thank you!

Once the ranking is confirmed, I will be the competition master! 🙂

### Overview

[image]

### Data

 I Created two types of extra data.

 1. binary label data. (toxic or nontoxic)
 2. more_toxic, less_toxic data

### CV strategy

[Jigsaw CV strategy](https://www.kaggle.com/its7171/jigsaw-cv-strategy)

### Models

|model|CV|loss|
|:---|:---:|:---:|
|microsoft/deberta-large|0.7032|BCE|
|gpt2-medium|0.6993|BCE|
|distilroberta-base|0.7012|MarginRankingLoss|
|microsoft/deberta-base|0.7043|MarginRankingLoss|
|xlnet-base-cased|0.6997|MarginRankingLoss|
|robelta-large|0.702|MarginRankingLoss|
|facebook/bart-base|0.7027|MarginRankingLoss|
|Hate-speech-CNERG/dehatebert-mono-english|0.6769|MarginRankingLoss|
|roberta-base|0.7017|MarginRankingLoss|
|microsoft/deberta-large|0.7014|MarginRankingLoss|

The larger model had a better CV.

max_len : 256

optimizer : AdamW
lr : 1e-5  
weight_decay : 0.1
epochs : 1~3
scheduler : get_linear_schedule_with_warmup

### Ensemble

I use optuna. As described below

```python
cols = ['less_toxic_pred', 'more_toxic_pred']
oof = []

oof.append(pd.read_csv('../input/jigsaw4-model2/bi003_deberta-large/oof.csv',usecols=cols).values)
oof.append(pd.read_csv('../input/jigsaw4-model1/exp008_roberta-base/oof.csv',usecols=cols).values)

def calc_score(weight):
    pred = np.zeros([n_data,2])
    for p, w in zip(oof,weight):
        pred += p*w
    score = (pred[:,0] < pred[:,1]).sum()/pred.shape[0]
    return score

class Objective:
    def __init__(self, n_models):
        self.n_models = n_models

    def __call__(self, trial):
        weight = [trial.suggest_uniform('weight' + str(n), 0, 1) for n in range(self.n_models)]
        return calc_score(weight)
    
max_iter = 1800
SEED = 29
objective = Objective(n_models)

sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(sampler = sampler,direction='maximize')
study.optimize(objective, n_trials = max_iter, n_jobs = -1)
```
