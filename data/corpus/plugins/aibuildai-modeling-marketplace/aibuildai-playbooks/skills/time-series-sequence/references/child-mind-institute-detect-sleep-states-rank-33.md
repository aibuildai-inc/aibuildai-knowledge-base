# 33th Place Solution - 1D-Unet with GRU, Detect repeating signals

Competition: child-mind-institute-detect-sleep-states
Rank: #33
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459632

First of all, we express our gratitude to the hosting team for organizing this intriguing competition.

This was my first chance to tackle the challenge of event detection, and I gained valuable insights from participating in this competition.

# Team Member
@sugupoko

# Overview and Summary

We utilized @tubotubo 's excellent notebook and code, implementing the following improvements. 
- Ensemble (Multi Duration, Multi Encoder Model)
- Insert BiGRU into 1D-Unet
- ​Parameter tuning for scipy.signal.find_peaks.
- Detect and repeating signals

I will explain these approaches in the order in which we tackled them in this competition.

# Ensemble
In this event detection task, we speculated that accurate predictions could be achieved by ensemble averaging the outputs of multiple models. Therefore, we applied the following ensemble approach:
### Multi Duration
We trained models with input durations of 8 hours (5760 steps), 16 hours (11520 steps), and 24 hours (17280 steps). During inference, we took the average of the outputs from these models."

### Multi Encoder Model
For the encoder model, we adopted timm-gernet-l, resnext101, and se-resnet50, and averaged the outputs of these three models, similar to the duration ensemble.

# Insert BiGRU into skip connection of 1D-Unet
With the idea from @sugupoko, we inserted BiGRU into the skip connections of the 1D-Unet decoder. We believe that this has improved the prediction accuracy in the temporal dimension.

# Parameter tuning for scipy.signal.find_peaks
While conducting several experiments, we noticed that the parameters of the `scipy.signal.find_peaks` function, used in post-processing, have an impact on accuracy. 
Initially, we lowered the `height` (score_th) to 0.001, as setting a lower threshold is advantageous in terms of the evaluation metric for this competition. Subsequently, we adjusted the `distance` to 65. This parameter determines the distance between peaks, and we are confident that setting an appropriate value helped reduce false positives.

# Detect repeating signals
While analyzing the data, we noticed that some input data contained unnatural periodic signals. In many cases, these signals lacked event labels and could potentially be a cause of false positives.
We didn't understand the reason for the presence of such signals, but we found that they occurred approximately every 24 hours (17280 steps). Therefore, we attempted to remove these repeating signals using a rule-based approach.
We created a function to detect repeating signals as follows. We segmented the anglez data at intervals of 17280 steps, calculating the correlation coefficient between the previous and current segments. If this correlation coefficient exceeded a certain threshold, it was determined to be a repeating signal. By invalidating the model output for the detected repeating segments, we were able to significantly reduce false positives.

```
def detect_repeated_area(series_id, cycle_length=17280, cor_th=0.8, phase='test'):​
    anglez = np.load(f"{CFG.processed_dir}/{phase}/{series_id}/anglez.npy")​
    ​
    valid_area = np.ones(anglez.shape[0])​
    x1_list = list(range(0, anglez.shape[0], cycle_length))​
    pre_anglez = np.zeros(cycle_length)​
    for x1 in x1_list:​
        x2 = x1 + cycle_length​
        cur_anglez = anglez[x1:x2]​
        ​
        if len(cur_anglez) < cycle_length:​
            cur_anglez = np.pad(cur_anglez, (0, cycle_length - len(cur_anglez)), constant_values=0.0)​
        ​
        correlation_coefficient_anglez = np.corrcoef(pre_anglez, cur_anglez)[0, 1]​
        ​
        if correlation_coefficient_anglez > cor_th:​
            valid_area[x1-cycle_length:x2] = 0​
        ​
        pre_anglez = cur_anglez​
        ​
    return valid_area
```


# Transition of LB Scores: 
- Baseline : 0.707​
- (5fold ensemble: 0.730)​
- Multi duration ensemble :0.745​
- Multi encoder model ensemble : 0.746​
- Insert BiGRU into 1D-Unet : 0.756​
- Parameter tuning for scipy.signal.find_peaks : 0.759​
- Detect repeating signals:0.764

# Things that didn't go well:
- Utilize sleep state prediction
- Ensemble longer and shorter durations
