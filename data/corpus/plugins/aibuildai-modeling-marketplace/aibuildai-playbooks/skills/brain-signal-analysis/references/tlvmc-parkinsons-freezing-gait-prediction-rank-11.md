# 11th place solution: LSTM-CNN + rolling features

Competition: tlvmc-parkinsons-freezing-gait-prediction
Rank: #11
Source: https://www.kaggle.com/c/tlvmc-parkinsons-freezing-gait-prediction/discussion/418008

# Features
I used the following features as model inputs, generated from AccV, AccML, AccAP

- Lags
- Global mean, median, max, average, std and quantiles
- Rolling mean, median, max, average, std and quantiles (+all of these applied to reversed time series)
- Rolling mean, median, max, average, std and quantiles of first difference of time series (+all of these applied to reversed time series)
- Mean number of sign changes over rolling window (+reversed time series)
- Exponentially weighted mean of first difference of time series
- Portion of time past

My code for the feature generation:
```python
def rolling_agg(
    dt: pd.DataFrame, step: int, 
    aggfunc: str, cols: list, back: bool = False) -> pd.DataFrame:
    
    if back:
        rolling = dt[cols][::-1].rolling(step, min_periods=0)
        suffix = f"_back_rolling_{step}_{aggfunc}"
    else:
        rolling = dt[cols].rolling(step, min_periods=0)
        suffix = f"_rolling_{step}_{aggfunc}"
        
    if aggfunc.startswith("quantile"):
        quantile = int(aggfunc.split("_")[1]) / 100
        
        return (
            rolling.quantile(quantile)
            .add_suffix(suffix))
    else:
        return (
            rolling.agg(aggfunc)
            .add_suffix(suffix))

def create_dataset(data, defog=False, verbose=False):
    cols = ['AccV', 'AccML', 'AccAP']
    dt = data.copy()
    
    if not defog:
        data[cols] = data[cols] / 9.80665
    dt["defog"] = int(defog)
    if verbose: print("Global stats")
    for aggfunc in ["mean", "max", "min", "std", "median"]:
        dt = dt.join(
            dt[cols].groupby(dt.assign(dummy=1).dummy)
            .transform(aggfunc).add_suffix(f"_{aggfunc}")
        )
    
    step1 = 500
    step2 = 100
    if verbose: print("Shifts stats")
    for shift in [1, 2, -1, -2]:
        
        if shift > 0:
            suffix_name = f"_lag_{shift}"
            fill_data = dt[cols].iloc[: shift]
        else:
            suffix_name = f"_lead_{abs(shift)}"
            fill_data = dt[cols].iloc[shift:]
        
        dt = dt.join(
            dt[cols]
            .shift(shift)
            .fillna(fill_data)
            .add_suffix(suffix_name)
        )
    
    aggfuncs = [
        "mean", "std", "max", "min", "median", 
        "quantile_75", "quantile_25", "quantile_99"
    ]
    if verbose: print("Rolling stats, step 1")

    for aggfunc in aggfuncs:
        
        dt = dt.join(
            rolling_agg(dt, step1, aggfunc, cols)
        )
    
    if verbose: print("Rolling stats, step 2")

    for aggfunc in aggfuncs:
        funcname = aggfunc if isinstance(aggfunc, str) else aggfunc.__name__
        dt = dt.join(
            rolling_agg(dt, step2, aggfunc, cols)
        )
    if verbose: print("Back Rolling stats, step 1")

    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(dt, step1, aggfunc, cols, back=True)
        )
    if verbose: print("Back Rolling stats, step 2")

    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(dt[::-1], step2, aggfunc, cols, back=True)
        )
    if verbose: print("Calculating diffs")

    diff = dt[cols].transform("diff").add_suffix("_diff")
    diff = diff.fillna(diff.iloc[0])
    cols = ["AccV_diff", "AccML_diff", "AccAP_diff"]
    if verbose: print("Diff rolling stat step 1")
    
    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(diff, step1, aggfunc, cols)
        )
        
    if verbose: print("Diff rolling stat step 2")
    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(diff, step2, aggfunc, cols)
        )
        
    if verbose: print("Back Diff rolling stat step 1")

    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(diff, step1, aggfunc, cols, back=True)
        )
    
    if verbose: print("Back Diff rolling stat step 2")

    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(diff, step2, aggfunc, cols, back=True)
        )
    
    if verbose: print("Sign change")
    sign_change = (
        diff.apply(np.sign)
        .transform("diff")
        .apply(np.abs)
        .divide(2)
        .fillna(0)
        .add_suffix("_sc")
    )
    
    cols = ["AccV_diff_sc", "AccML_diff_sc", "AccAP_diff_sc"]
    aggfuncs = ["mean"]
    if verbose: print("Sign change rolling stat step 1")

    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(sign_change, step1, aggfunc, cols)
        )
    if verbose: print("Sign change rolling stat step 2")

    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(sign_change, step2, aggfunc, cols)
        )
    if verbose: print("Back Sign change rolling stat step 1")

    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(sign_change, step1, aggfunc, cols, back=True)
        )
    if verbose: print("Back Sign change rolling stat step 2")

        
    for aggfunc in aggfuncs:
        dt = dt.join(
            rolling_agg(sign_change, step2, aggfunc, cols, back=True)
        )
        
    if verbose: print("time spent")
    
    dt["time_spent"] = dt.Time.divide(dt.Time.max())
    return dt.drop("Time", axis=1).fillna(0)
```

After those transformations, I scaled data and divided all time series into parts of length 10000 in separate files of feather format

# Model

I used LSTM-CNN model: the input is first fed into three parallel blocks of Conv1D with the different kernel sizes of 3, 5 and 7. Then the input is concatenated with the output of those conv layers and passed to two sequential layers of LSTM. After all - there is a linear layer that does the classification

Here's a code for model definition

```python
import torch.nn as nn
import torch.nn.functional as F

def block(kernel_size):
    return nn.Sequential(
        nn.Conv1d(240, 128, kernel_size, padding="same"),
        nn.ReLU(),
        nn.Conv1d(128, 64, kernel_size, padding="same"),
        nn.ReLU(),
    )

class ParkinsonModel(nn.Module):
    def __init__(self, kernels=[3, 5, 7]):
        self.kernels = kernels
        super().__init__()
        self.conv_nets = nn.ModuleList([
            block(i) for i in kernels
        ])
        self.lstm = nn.LSTM(
            64 * len(self.kernels) + 240, 
            128, 2, batch_first=True, bidirectional=True, dropout=.1)
        self.linear = nn.Linear(128 * 2, 4)

    def forward(self, x):
        conv_res = []
        for net in self.conv_nets:
            conv_res.append(net(x))
        conv_res.append(x)
            
        conv_res_tensor = torch.concat(conv_res, axis=1)
        lstm_out, _ = self.lstm(conv_res_tensor.transpose(2, 1))
        res = self.linear(lstm_out).transpose(2, 1)
        return res
```

I did not manage to create a good validation pipeline, so i didn't use any folds. I trained the model for 30 epoch, monitoring loss on validation data (approx. 10% of subjects)
