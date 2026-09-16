# #43 Solution | Catboost + RealMLP

Competition: playground-series-s6e2
Rank: #43
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/43-solution-catboost-realmlp

Thanks to the Kaggle team for hosting another Tabular Playground competition. It's been a while since I've been this engaged in a competition. Halfway through, I discovered a new model I hadn't used before, which reignited my interest.

# Initial Approach
As usual, I began by running a series of experiments using traditional models.

| Model | # of Models | Best local CV score |
| --- | --- | --- |
| HGB | 30 | 0.95524 |
| LGBM | 50 | 0.95435 |
| XGB | 50 | 0.95426 |
| CatBoost | 30 | 0.95555 |

I soon realized that `CatBoost` was performing well on this dataset without much tuning. In particular, my best `CatBoost` model used `max_depth=3`.

# Feature Engineering
I engineered meta-features from the original dataset and appended them to the competition dataset, as many participants did. I also considered target encoding. At this point, I decided to focus on `LGBM`, `XGB`, and `CatBoost`.

| Model | Best local CV score |
| --- | --- |
| XGB + meta features | 0.95446 |
| XGB + target encoding | 0.95444 |
| XGB + meta features + target encoding | 0.95546 |
| LGBM + meta features | 0.95547 |
| LGBM + target encoding | 0.95549 |
| LGBM + meta features + target encoding | 0.95549 |
| CatBoost + meta features | 0.95557 |
| CatBoost + target encoding | 0.95551 |
| CatBoost + meta features + target encoding | 0.95552 |

# Ensemble 
My best ensemble combined five different versions of `CatBoost` with `Ridge` as the meta-learner (CV = 0.95560). I also experimented with other meta-learners, such as Logistic Regression, and tried hill-climbing as an ensemble technique.

# Adding RealMLP 
My typical approach in these month-long competitions is to spend the first two weeks running a variety of experiments without consulting public notebooks or discussions. In the second half, I begin exploring public notebooks and ideas shared in the discussion sections. During this phase, I discovered this [notebook](https://www.kaggle.com/code/omidbaghchehsaraei/the-best-solo-model-so-far-realmlp-lb-0-95397) featuring RealMLP, a model I hadn't encountered before, which outperformed my best ensemble. I integrated two versions of RealMLP into my ensemble with `Ridge` as the meta-learner, achieving a local CV of 0.95572, a public LB of 0.95394, and a private LB of 0.95532.
