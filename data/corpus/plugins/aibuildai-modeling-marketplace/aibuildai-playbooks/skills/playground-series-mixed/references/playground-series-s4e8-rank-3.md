# [1st Place Solution AutoML Grand Prix] AutoML Grandmasters: AutoGluon Distributed + Post-Hoc Ensembling

Competition: playground-series-s4e8
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s4e8/discussion/523656

Heyho everyone!

Another exciting 24 hours of automatically fitting machine learning models! Kudos to all the competitors, and a shoutout to Team AGA, who gave us a run for our money with an early `0.98533` score that we spent the next 12 hours trying to beat. In the end, the winner was decided by a difference in score so small that Kaggle’s leaderboard doesn’t even visualize it, so we consider ourselves pretty lucky to have eeked it out. We detail our approach to the fourth Grand Prix in the following. 

But first, we would like to highlight that **the International Conference on Automated Machine Learning 2024 is is taking place in Paris from September 09. to 12.** (see https://2024.automl.cc/)! We are going to be at the conference and would be happy to meet other (Auto)ML enthusiasts from Kaggle at the conference :) 

---

## Overview Figure
[SolutionOverview]

---

## Summary 

We used [AutoGluon](https://auto.gluon.ai/) as our AutoML system, installing via source on the latest mainline with tweaks to enable distributed computation (see below). With it, our approach to this competition can be summarized as follows:

#### Preprocessing
1. We replaced noisy observations in the data. Nothing else improved our offline cross-validation (CV) score. 

#### Model Fitting 

We ran a default version of AutoGluon for one hour and a custom version of AutoGluon for four hours. For the custom version, we used the following settings:

1. We used log loss as an early stopping metric while optimizing/picking based on the target metric MCC. We are unsure if this helped, but it improved our offline CV score.
2. We used 16-fold cross-validation and AutoGluon’s multi-layer [stacking](https://auto.gluon.ai/dev/tutorials/tabular/how-it-works.html) implementation.
3. We trained a customized portfolio of models, meta-learned from [TabRepo](https://github.com/autogluon/tabrepo) (i.e., zero-shot HPO)
4. We used 100 iterations (instead of the default 25) for [*post hoc* ensembling](https://arxiv.org/abs/2307.08364). 

#### Kaggle Tricks

1. AutoGluon rounds to the first 6 decimals of the score when determining tiebreakers during the final weighted ensemble. It turns out that is too few for the deltas we were looking for, so we upped it to 8 at the very end to eke out an epsilon improvement.
2. We manually created a *post hoc* ensembling logic for this competition in an effort to go from `0.98531` to `0.98533`. We cached all pred probas from all models on all experiments and then ran the *post hoc* ensembling for the final solution. This allowed us to just barely leapfrog Team AGA!

#### Compute

After the last competition, we felt the need to expand our computing power for these (very) large data competitions. Therefore, in addition to using an individual AWS compute instance, we put much effort into using AutoGluon distributed across compute nodes. 

1. We used a prototype of a distributed version of AutoGluon to parallelize AutoGluon across compute resources.
2.  We used an in-house SLURM cluster (from the University of Freiburg) together with [Ray](https://docs.ray.io/en/latest/index.html) to distribute AutoGluon’s model training across 1000 CPUs. 

---

## Code

The supplementary code repository for this write-up can be found here: https://github.com/AutoML-Grandmasters/Fourth-AutoML-Grand-Prix/tree/main

Our copy-able settings for the custom run of AutoGluon can be found in this [file](https://github.com/AutoML-Grandmasters/Fourth-AutoML-Grand-Prix/blob/main/main.py).

### Preprocessing
We used the following code to clean noisy observations that do not exist in the test predictions by setting them to nan. 

```python

import numpy as np
import pandas as pd

train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

weird_columns = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "gill-attachment",
    "gill-spacing",
    "gill-color",
    "veil-type",
    "veil-color",
    "has-ring",
    "ring-type",
    "spore-print-color",
    "habitat",
    "does-bruise-or-bleed",
    "stem-root",
    "stem-surface",
    "stem-color",
]

for col in weird_columns:
    allowed_vals = test_data[col].unique()
    train_data.loc[~train_data[col].isin(allowed_vals), col] = np.nan
    test_data.loc[~test_data[col].isin(allowed_vals), col] = np.nan
```

### Early Stopping Metric 

To early stop on `log_loss`, pass the following to AutoGluon’s fit call: `ag_args_fit={"stopping_metric": "log_loss"}`. This can sometimes help as early stopping on threshold-based metrics such as MCC can be too early. 

### Distributed AutoGluon & Compute

Here is the current version of distributed AutoGluon: https://github.com/LennartPurucker/autogluon/tree/distributed_autogluon
An example script with more details on how to use it can be found here: https://github.com/AutoML-Grandmasters/Fourth-AutoML-Grand-Prix/blob/main/autogluon_distributed_example.py

In our experience with it, it is mostly stable but has a few GPU-related problems that we hope to fix for the prototype. We plan to integrate a more mature version of this into mainline AutoGluon in the future (and are already working on it). 

We used 1000 CPUs spread across nodes with 20 or 32 Intel(R) Xeon(R) Gold 6242 CPUs @ 2.80GHz, each node with around 150 GB of RAM. The cluster is managed with [SLURM](https://slurm.schedmd.com/documentation.html), and we used [Ray](https://docs.ray.io/en/latest/index.html) to create a sub-cluster that AutoGluon can (natively) use to fit models. To run AutoGluon distributed, only a Ray cluster is needed, which can be created on local compute, SLURM clusters, or cloud resources.  

Additionally, we used an AWS m7i.48xlarge EC2 instance with 192 vCPUs to run default AutoGluon.


### Customized Portfolio 

To obtain a better portfolio than currently existing in AutoGluon, we re-ran the work of the [TabRepo paper](https://arxiv.org/pdf/2311.02971) (to be presented at the AutoML conference), but use a portfolio of size 200 instead of the 100 size portfolio used by AutoGluon’s best_quality setting. We also included more model families: Linear models and KNN, although some of those models we ended up disabling those since they weren’t helping and, at times, took a long time to infer. 

We additionally added configurations with larger `max_bin` values for LightGBM, XGBoost, and CatBoost. Moreover, we removed a set of configurations that we found to take too long to predict for large datasets. You can find the final portfolio in this [file](https://github.com/AutoML-Grandmasters/Fourth-AutoML-Grand-Prix/blob/main/tabrepo_2024_custom.py). We filtered the portfolio to models that were working well (see [here](https://github.com/AutoML-Grandmasters/Fourth-AutoML-Grand-Prix/blob/main/main.py#L69-L84)).

### More Iterations for Post Hoc Ensembling 

We wanted to find a better final post hoc ensemble by giving the [greedy ensemble selection](https://dl.acm.org/doi/10.1145/1015330.1015432) in AutoGluon more iterations. Sadly, so far, there is no easy interface to increase this number. Thus, we simply monkey patched our local install of AutoGluon. To do so, one needs to set the value in [this line](https://github.com/autogluon/autogluon/blob/master/core/src/autogluon/core/models/greedy_ensemble/greedy_weighted_ensemble_model.py#L22) to 100.

### Kaggle Tricks

* To adjust the number of decimals for rounding, set the `round_decimals` variable in
`autogluon.core.models.greedy_ensemble.ensemble_selection.EnsembleSelection` Line 112 to `8` (see [here](https://github.com/autogluon/autogluon/blob/f5c7b7400596e1449fad2b4107fde55809a09bac/core/src/autogluon/core/models/greedy_ensemble/ensemble_selection.py#L112) on GitHub).
* Our manual post hoc ensembling logic can be found in this [file](https://github.com/AutoML-Grandmasters/Fourth-AutoML-Grand-Prix/blob/main/ag_post_hoc_ensembler.py). It assumes that you have a finished run of AutoGluon on disk (in our cases we used the default one-hour run for this) and the following artfiacts from another run of AutoGluon: a) the [prediction probabilities](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.predict_proba.html) on test data, b) the [out-of-fold prediction probabilities](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.predict_proba.html) on training data, and c) [predictions](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.predict.html) on test data. 


---

Best regards,
Lennart, Nick (@innixma), and Arjun (@neonkraft), on behalf of the "AutoML Grandmasters"

Prior Grand Prix Competition Write-ups: [First AutoML Grand Prix Competition](https://www.kaggle.com/competitions/playground-series-s4e5/discussion/500783), [Second AutoML Grand Prix Competition](https://www.kaggle.com/competitions/playground-series-s4e6/discussion/509631), [Third AutoML Grand Prix Competition](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/516265)
