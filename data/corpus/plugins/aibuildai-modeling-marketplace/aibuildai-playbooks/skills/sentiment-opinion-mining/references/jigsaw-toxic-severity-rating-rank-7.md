# 7th Place Solution

Competition: jigsaw-toxic-severity-rating
Rank: #7
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306366

Our team didn't trust public LB, and worked towards improving the score on the un-processed validation_df. In the end, we improved the score to 0.7211 without leaks, which was the 7th best score in private LB. We thought this was a great result because we kept running Trust CV, but we later realized that we had ensembled models that had won the lottery.

As other teams have pointed out, the private LB dataset's test comments overlapped with jigsaw's first competition, the Toxic Comment Classification Challenge. Therefore, it was possible to create a model within the gold zone by training this dataset directly.

Our team placed in the gold zone, partly because we did a Trust CV, but essentially because we had models that won the lottery and we gave them enough weight in the weighted average. 

Here's a brief overview of what we've been working on since.

# Model × Dataset

The following table shows the models used for the final submission, the dataset trained, and the scores. The "pseudo label" refers to the model trained with validation_df in toxic-xlm-roberta and pseudo-labeled on the dataset. This pseudo-labeling greatly improved the CV, while private LB did not.

In addition, we strongly emphasized the importance of not leaking when training. Specifically, when training validation_data, union-find GroupKFold was used, and when training other datasets, the text in validation_data was removed.

| name | model | dataset | CV | PublicLB | PrivateLB |
|---|---|---|---|---|---|
| mst029 | toxic-xlm-roberta | Ruddit (Pseudo Label) | 0.7126 | 0.77391 | 0.79796 |
| mst030 | toxic-xlm-roberta | jigsaw 1st (Pseudo Label) | 0.7224 | 0.77587 | 0.79676 |
| msttweet | toxic-xlm-roberta | Toxic Tweet (Pseudo Label) | 0.7155 | 0.77587 | 0.79641 |
| colum014 | roberta-base | validation_df | 0.70307 | 0.79590 | 0.79410 |
| colum015 | roberta-large | validation_df | 0.70307 | 0.80189 | 0.79744 |
| colum016 | deberta-v3-base | validation_df | 0.70646 | 0.78730 | 0.79594 |
| colum018 | unbiased-toxic-roberta | validation_df | 0.70426 | 0.80341 | 0.79416 |
| colum019 | toxic-xlm-roberta | validation_df | 0.70898 | 0.76749 | 0.79644 |
| colum020 | toxic-bert | validation_df | 0.70456 | 0.77533 | 0.79422 |
| calpis001 | roberta-large | jigsaw 1st | 0.7031 | 0.79111 | 0.81814 |
| calpis011 | toxic-xlm-roberta | jigsaw 1st (Pseudo Label) | 0.7205 | 0.77152 | 0.79532 |
| calpis012 | toxic-xlm-roberta | Toxic Tweet (Pseudo Label) | 0.7141 | 0.76967 | 0.79743 |
| naoism1004 | roberta-large | jigsaw 1st | 0.7048 | 0.80581 | 0.81520 |
| naoism4001 | toxic-xlm-roberta | jigsaw 1st (Pseudo Label) | 0.7220 | 0.76325 | 0.79824 |
| naoism4003 | toxic-xlm-roberta | Civil Comment Dataset (Pseudo Label) | 0.7103 | 0.76249 | 0.79584 |

# Ensemble

We used these models to perform the ensemble of weighted averages. The weights were determined using Optuna, which was implemented by @calpis10000 .

The implementation is as follows. We used the array of less and more predictions corresponding to validation_df to automatically determine the weights for each model that would score best. Also, there was a problem that the scales of the predictions for each model were different in this case. Therefore, we tried MinMaxScaler, StandardScaler, RankGauss, and so on. The best one was StandardScaler.

```python
def val(trial):
    p1 = np.zeros(30108)
    p2 = np.zeros(30108)
    
    x0 = trial.suggest_float(pred_columns[0], 0, 1)
    x1 = trial.suggest_float(pred_columns[1], 0, 1)
    x2 = trial.suggest_float(pred_columns[2], 0, 1)
    x3 = trial.suggest_float(pred_columns[3], 0, 1)
    x4 = trial.suggest_float(pred_columns[4], 0, 1)
    x5 = trial.suggest_float(pred_columns[5], 0, 1)
    x6 = trial.suggest_float(pred_columns[6], 0, 1)
    x7 = trial.suggest_float(pred_columns[7], 0, 1)
    x8 = trial.suggest_float(pred_columns[8], 0, 1)
    x9 = trial.suggest_float(pred_columns[9], 0, 1)
    x10 = trial.suggest_float(pred_columns[10], 0, 1)
    x11 = trial.suggest_float(pred_columns[11], 0, 1)
    x12 = trial.suggest_float(pred_columns[12], 0, 1)
    x13 = trial.suggest_float(pred_columns[13], 0, 1)
    x14 = trial.suggest_float(pred_columns[14], 0, 1)

    x_list = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]
    for i, x in enumerate(x_list):
        p1 += scaler_less[:, i] * x
        p2 += scaler_more[:, i] * x
    
    score = (p1 < p2).mean()
    return score

optuna.logging.disable_default_handler()
study = optuna.create_study(
    direction='maximize', 
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(val, n_trials=1000)

display(study.best_trial.number)
display(study.best_trial.values)
display(study.best_trial.params)
```

The weights are as follows. It was good to see that much greater weight was given to calpis001 and naoism1004. The results were such that highly correlated predictions didn't give as much weight to each other, which was consistent with our goal of finding a weighted average that would produce diversity.

```
{
 'mst029_toxic': 0.14599159334648742,
 'mst030_toxic': 0.9400258577765264,
 'msttweet_toxic': 0.2296755364508963,
 'colum014_toxic': 0.46435656956718,
 'colum015_toxic': 0.8156998001053446,
 'colum016_toxic': 0.6003326557886719,
 'colum018_toxic': 0.08731982956387017,
 'colum019_toxic': 0.02050220970508744,
 'colum020_toxic': 0.8002815649764881,
 'calpis001_toxic': 0.8907374041352457,
 'calpis011_toxic': 0.46794326158461463,
 'calpis012_toxic': 0.0613928819716386,
 'naoism1004_toxic': 0.9998908148400331,
 'naoism4001_toxic': 0.9085892865154106,
 'naoism4003_toxic': 0.0312185722129308
}
```

The above is our solution. Thank you @mst8823 , @calpis10000 , and @naoism for teaming up with us!

Inference code: [[Jigsaw] team-ensemble 006 fix](https://www.kaggle.com/columbia2131/jigsaw-team-ensemble-006-fix/notebook)
