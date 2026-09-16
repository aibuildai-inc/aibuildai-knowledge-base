# Blending best public submissions - 17th | Simple approach - ~70th

Competition: playground-series-s3e19
Rank: #17
Source: https://www.kaggle.com/c/playground-series-s3e19/discussion/428433

Hello all,
I want to show you my simple solution to this competition which results in about 70th place and blending the best public submissions, which allows us to get 17th place.

## 17th Solution:
The 17th solution is simple. I've noticed that @yeoyunsianggeremie published a dataset with the best public submissions, available here: https://www.kaggle.com/datasets/yeoyunsianggeremie/s3e19-top-public-notebook-submissions. Since @paddykb mentioned the probable data construction error, I've concluded that probing the LB may work in this competition. Therefore, I was just curious about what we can accomplish by blending these best public predictions with mean strategy. As it turned out, the result allowed us to gain the 17th result of private LB (top 2%). Okay, so there is no analysis, just blend these best public submissions and assuming there is a data construction error, it should work fine. The complete code snippet is available below:
```
import glob
from functools import partial

import pandas as pd
import numpy as np


path = "/kaggle/input/s3e19-top-public-notebook-submissions"

best_public_lb_paths = glob.glob(f"{path}/*.csv")
best_lbs = pd.concat(map(partial(pd.read_csv, index_col="id"), best_public_lb_paths), axis=1)
best_lbs.columns = [name.split("/")[-1] for name in best_public_lb_paths]
best_of_best_lbs = best_lbs.drop(  # Several submissions are not postprocessed.
    [
        "bogoconic1_48.86.csv",
        "nivedithavudayagiri_38.87.csv",
        "oscar_no_postprocessing.csv",
        "paddykb_no_postprocessing.csv",
    ],
    axis=1,
)

submission = pd.DataFrame(
    {
        "id": best_of_best_lbs.index,
        "num_sold": best_of_best_lbs.mean(axis=1).astype(np.int32),
    }
).set_index("id")

submission.to_csv("submission.csv")
```

## ~70th Solution:
So, let's get to my original work, which resulted in about 70th on private LB (top 6%). We also don't have any rocket science here, and the approach is as simple as possible. Preprocessing supposes several steps as follows:
- Fixing the Covid period - There was a small breakdown during the Covid-19 lockdown, which is fine to resolve. Generally, I used a simple strategy of multiplying each observation by a certain factor from the beginning of March 2020 up to the end of July 2020. I've just taken a mean number of sold products for each observation from this period in the years 2017, 2018 and 2019. This way, we obtain factors by dividing these mean values by values observed during the Covid lockdown.
- Numeric Dates - I added a numeric date version to the dataset. These are obviously `Day`, `Month`, `Year`, and flags if there is a weekend - `IsWeekend`, and Sunday - `IsSunday`.
- Trigonometric Features - I added sine functions, i.e. `MonthSin` and `DaySin`.
- Holidays - I included a holiday flag, i.e. `IsHoliday`, for each country in the dataset using the `holiday` library.
- Original Features - Just encode these with `OrdinalEncoder`. 

Finally, the model is just an ordinary `LGBMRegressor` with a little bit of regularization found by `Optuna`. Additionally, I fit it to the logarithmic version of the target. The parameters are as follows:
```
params = {
    "random_state": 51,
    "learning_rate": 0.17,
    "min_child_samples": 336,
    "colsample_bytree": 0.5,
    "reg_lambda": 8.35,
    "reg_alpha": 0.36,
}
```
Since the public LB score obtained with that model was horrible, I started to probe, which has led to the following multipliers:
```
probed_map = {
    "Argentina": 4.5,
    "Spain": 1.6,
    "Japan": 1.2,
    "Estonia": 1.7,
    "Canada": 0.9,
}
```
resulting in 6.08955 on public LB and 7.21200 on private LB. Since I joined the competition a couple of days before it finished, I couldn't boost this result any more, but I think it's decent regarding the simplicity of the approach. In fact, probably to get this result, we don't need to bother with the model and only find the coefficients.

Many thanks to:
- @yeoyunsianggeremie - for publishing dataset with best public submissions.
- @paddykb - for this post: [Data construction error?](https://www.kaggle.com/competitions/playground-series-s3e19/discussion/425538)
- @ravi20076 - for this post: [Request for some variety in ongoing challenges](https://www.kaggle.com/competitions/playground-series-s3e17/discussion/417785), and discussions.

**My notebook is available here if somebody is interested: [Playground Series S3E19 - Forecasting Sales](https://www.kaggle.com/code/mateuszk013/playground-series-s3e19-forecasting-sales)**

Have a nice day!
