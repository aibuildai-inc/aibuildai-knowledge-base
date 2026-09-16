# Third place solution: mode ensemble of 6 public notebooks.

Competition: playground-series-s3e5
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e5/discussion/386683

Closing time! Very surprised to have survived the shake-up.

[As I hinted in a previous thread.](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/384925). A simple ensembling strategy takes a high public LB spot, and now it appears this strategy also takes a high private LB spot ;).

I used the following notebooks:
1) [PS-3-5 Keras NN Model](https://www.kaggle.com/code/martynovandrey/ps-3-5-keras-nn-model/data?select=submission.csv) by @martynovandrey
2) [PS-S3E5 using Polars](https://www.kaggle.com/code/kotrying/ps-s3e5-using-polars/data) by @kotrying
3) [PS s3e5 - NN & LightAutoML & Thresholding, Oh my!](https://www.kaggle.com/code/paddykb/ps-s3e5-nn-lightautoml-thresholding-oh-my/data?select=submission_lightautoml_cv_0.562.csv) by @paddykb
4) [SR with inequalities -> simple model](https://www.kaggle.com/code/jano123/sr-with-inequalities-simple-model/data) by @jano123
5) [S3E5 - 0.624 XGB/LGB + Kmeans & PCA](https://www.kaggle.com/code/usedpython/s3e5-0-624-xgb-lgb-kmeans-pca/data) by @usedpython
6) [PS s3e5 - Repeated SR with inequalities](https://www.kaggle.com/code/paddykb/ps-s3e5-repeated-sr-with-inequalities/data) by @paddykb

I then used a simple weighted mode on these submissions, in which I added 6) two times to the list (so double weight). The code is below and I published a [notebook with this code (although LB scores are somewhat different due to version mismatches)](https://www.kaggle.com/group16/mode-ensemble-3-private)

```python
import pandas as pd
import numpy as np
import glob
from sklearn.utils.extmath import weighted_mode

subs = []
for i, file in enumerate(sorted(glob.glob('subs/*'))):
    sub_df = pd.read_csv(file)
    print(sub_df.head(5))
    sub_df = sub_df.rename(columns={'quality': f'quality_{i}', 'id': 'Id'})
    sub_df = sub_df.sort_values('Id')
    sub_df = sub_df.set_index('Id', drop=True)
    subs.append(sub_df)
    
# Add sub6.csv twice to the list for double weight
dup_sub = subs[5]
dup_sub = dup_sub.rename(columns={'quality_5': 'quality_6'})
subs.append(dup_sub)

all_sub_df = pd.concat(subs, axis=1)

# Manually make these class weight with small epsilon multiplied by index in counts
class_weights = {
    3: 1.01,
    4: 1.03,
    5: 1.06,
    6: 1.05,
    7: 1.04, 
    8: 1.02
}

def my_weighted_mode(x):
    values = x.values
    weights = [class_weights[i] for i in values]
    
    return weighted_mode(values, weights)[0][0]

sub = subs[0].copy()
sub = sub.drop(columns=['quality_0'])
sub['quality'] = all_sub_df[[f'quality_{i}' for i in range(len(subs))]].apply(my_weighted_mode, axis=1).astype(int)
sub
```

It turns out that the weighted mode might be a bit of an overkill here and `scipy.stats.mode` has the same results.

I selected the second-best scoring sub on the private LB as one of my final subs. One of my subs could have been #1, but that is always easy to say in hindsight! I am happy to make the top-3 and get a piece of exclusive Kaggle swag :).


