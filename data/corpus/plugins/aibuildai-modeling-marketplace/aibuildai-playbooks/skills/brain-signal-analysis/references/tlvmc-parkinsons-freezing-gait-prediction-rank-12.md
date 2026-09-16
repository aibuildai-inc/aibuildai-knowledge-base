# 12th place solution: Simple features and LSTM.

Competition: tlvmc-parkinsons-freezing-gait-prediction
Rank: #12
Source: https://www.kaggle.com/c/tlvmc-parkinsons-freezing-gait-prediction/discussion/416248

I would like to thank host and kaggle for organizing such interesting competition.
I believe that the method proposed by the winner and prizewinners who received high scores in both the public and private tests will be helpful to host.

## Summary
- Ensamble of 4 model based by LSTM
- Use 4 features. mean, std, max, min, median.
- Notype data is used as validation data.
- Selected model separately for tDCS FOG and DeFOG.

## Data Preprocessing
I compressed all data into fixed length. Length is 2048 because shortest training data is 2359.

Compression Method
1. Expand the original data to a smallest multiple of the target length that exceeds the original data length. (ch, target_size * n)
2. Fold the data into (ch, target_size, n).

As feature, I used the mean, std, max, min, and median within this N.
In addition to this, it was also useful to add each percentile point (15~90 percentile).
I rarely used the information on each ID and Subjects, but only "Visit" was used as a label for multitasking.

```python
def resize_func(x, target_size=2048, use_percentile_feat=False):
    ch = x.shape[0] 
    input_size = x.shape[1]
    
    pad = target_size - input_size % target_size
    factor = (input_size + pad) / input_size

    x = np.array([ndi.zoom(xi, zoom=factor, mode='reflect') for xi in x])
    x = x.reshape((ch, target_size, -1))

    res = {} 
    res['mean'] = np.mean(x, axis=2).reshape(ch, -1)
    res['max'] = np.max(x, axis=2).reshape(ch, -1)
    res['min'] = np.min(x, axis=2).reshape(ch, -1)
    res['med'] = np.median(x, axis=2).reshape(ch, -1)
    res['std'] = np.sqrt(np.var(x, axis=2).reshape(ch, -1))
    if use_percentile_feat:
        for p in [15, 30, 45, 60, 75, 90]:
            res[f"p{p}"] = np.percentile(x, [p], axis=2).reshape(ch, -1)

    return res
```

## Model
The model used was LSTM: a simple 4-block model and a model that were [3rd place of similar competitions in the past](https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/285330). For DeFOG, I also used [Transformer](https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-112).


## CV Strategy
- tDCS FOG
In tDCS FOG, I used 4fold StratifiedGroupKfold with Event label and Subject as group.
However, CV was quite high for certain folds.
This is probably due to the large number of examples of difficult classes such as StartHesitation and Walking appearing for a long time in Subejt "2d57c2", which reduced the percentage of False Positives.
Therefore, I had selected a model based on a score that excluded this Subject.
Personally, I think that the reason why Shake occurred was also due to the existence of such a Subject in the Public data, and that many teams overfit to it.

- DeFOG
 Because DeFOG has few labeled data, I used all labeled data as training data.
Since there was a correlation between the loss of the three labels and the loss of the Event label only, I used Nototype as the validation data.
Although it was not possible to make a final selection, I trained on all data including tdcsfog, and the model selected based on the Notype data was the my best private score([this notebook](https://www.kaggle.com/code/mrt0933/keras-lstm-trained-by-all-labeled-data)).

## Final submission
I selected 4 models for each of tdcsfog and defog and ensemble them.
For tDCS FOG, I chose the one with the highest cv for each fold.
The best public score had a high enough cv, so I selected this score as the final submission (I'm glad I didn't get caught in the shaking).

- tDCS FOG. CV:0.28

| Fold | Model | CV |
|:-----------|------------:|:------------:|
| 0 (w/o "2d57c2”) |  LSTM (multitasking) | 0.179  |
| 1 |  LSTM (1d CNN head) | 0.337  |
| 2 |  LSTM (add percentile features) | 0.278 |
| 3 |  LSTM (1d CNN head) | 0.293 |

- DeFOG. CV: 0.381(Event)

| No | Model | CV |
|:-----------|------------:|:------------:|
| 0 |  LSTM | 0.292  |
| 1 |  LSTM (add percentile features) | 0.318  |
| 2 |  LSTM (long 4096) | 0.286 |
| 3 |  Transformer | 0.286 |

Public LB: 0.444
Private LB: 0.341


Finally, I am happy to become a kaggle master with this gold medal.
However, although I got a gold medal, I am disappointed that I could not design a model that could capture FOG events well both quantitatively and qualitatively.
I would love to participate in the second and third competitions if they are held. Thank you very much.
