# 20th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #20
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369538

Thanks to organizers for this competition and all the competitors!
Here is a brief introduction of my solution with the highest Private LB score (equivalent to 20th place).

# Overview

My solution consists of a weighted average of several BERT models using the Nelder-Mead method and pseudo-labeling with text data from feedbacks 1 and 2.

[**my inference code**](https://www.kaggle.com/code/columbia2131/fb3-sub-exp16202122303132333638464748)

# What Worked

### High Impact Idea

**Pseudo-Labeling**

I trained deberta-v3-large on this competition data and then  pseudo-labeled on the Feedback 1 and 2 data, excluding data that overlapped with this competition.  
The pseudo-labeled data was trained by deberta-v3-base, deberta-v3-large and deberta-v2-xlarge following [a meta pseudo labels method](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315707#:~:text=org/abs/2003.10580-,Meta%20Pseudo%20Labels,-notebook%20link%20on)(CV: +0.005~0.01).

**Nelder-Mead method**

The welder-mead method is one of the algorithms for optimization problems and is implemented in the minimize function of scipy.  
The optimal weighted average method using this technique is based on [2nd solution method in the CommonLit Competion](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328) by @takoihiraokazu.

The implementation is in [my solution](https://www.kaggle.com/code/columbia2131/fb3-sub-exp16202122303132333638464748), which is as follows:
```python
from scipy.optimize import minimize

target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

def optimeze_func(weights, idx=0, col='cohesion'):
    opt_preds = np.zeros(len(train))
    for i in range(len(weights)):
        opt_preds += oof_preds[i, :, idx] * weights[i]
    score = mean_squared_error(train[col], opt_preds, squared=False)
    return score

opt_oof_preds = np.zeros((len(train), len(target_cols)))
opt_sub_preds = np.zeros((len(test), len(target_cols)))
weight_df = []

for idx, col in enumerate(target_cols):
    func = lambda x: optimeze_func(x, idx, col)
    weights = [1 / len(oof_preds)] * len(oof_preds)
    result = minimize(func, weights,  method="nelder-mead")
    weight_df.append(result.x)
    for i, weight in enumerate(result.x):
        opt_oof_preds[:, idx] += oof_preds[i, :, idx] * weight
        opt_sub_preds[:, idx] += sub_preds[i, :, idx] * weight

```

### Medium Impact Idea

**Ensemble of multiple BERT models**

The following model was used:
* deberta-v3-base
* deberta-v3-large
* deberta-large
* muppet-roberta-large
* roberta-base
* roberta-large
* funnel-transformer-large
* funnel-transformer-xlarge
* deberta-v2-xlarge

The learning parameters for all models are as follows:
* max_length = 512
* criterion = nn.SmoothL1Loss(beta=1., reduction='mean')
* learning_rate = 1e-5
* optimizer = AdamW(betas=(0.9, 0.98), weight_decay=2e-5)
* scheduler = get_linear_schedule_with_warmup
* num_warmup_steps_rate = 0.01
* clip_grad_norm = 1000
* all dropout = 0

# What Didn't Worked

* Stacking with GBDT

### Methods that could have been improved in accuracy if worked on

* Adversarial Weight Perturbation(AWP)
* Layerwise Learning Rate(LLR)
* Support Vector Regression(SVR) trained using embedding vectors of pre-trained models

etc.

# Important Citations:

* [FB3 / Deberta-v3-base baseline [train]](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train) by @yasufuminakama
* [2nd place solution](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328) in CommonLit competition by @takoihiraokazu
