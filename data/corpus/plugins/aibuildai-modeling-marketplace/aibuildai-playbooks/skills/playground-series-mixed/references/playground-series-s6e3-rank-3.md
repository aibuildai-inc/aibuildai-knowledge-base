# 3rd Place Solution - An Ensemble of 100 OOFs

Competition: playground-series-s6e3
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s6e3/writeups/3rd-place-solution-an-ensemble-of-100-oofs

I'm super excited to get a 3rd place finish this month!
This month, unlike others, I really focused on ensembling, rather than one strong model.
Every day I would write down a list of new ideas to try and run models both locally and via Kaggle notebooks.

I didn't have as much time this month so I used a combination of Generative AI/my own work/editing public kernels.

In the end I used 100 OOF files - a mix of my own models and models derived from public kernels (see references below).

### Models and Ensembling

[Ensemble pipeline]

#### XGB

I started this month with an XGB notebook (https://www.kaggle.com/code/include4eto/single-xgb-cudf-pseudo-labels-cv-0-91789) with some blanket feature engineering from last month's competition.

This wasn't the best model by far, but a good starting point and skeleton for writing other models.

I wrote many variations of this model with different feature combinations and tree parameters.
This is definitely a good way to add variety to your ensemble.

#### Local Notebooks

The "local" notebooks references below were models run on my machine.

A very interesting model (due to its simplicity) is discussed here (https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679983).
A simple model of `depth=2` and all features as categorical mixed well with the rest of my ensemble (`025_md_v2`).

Other than that I experimented locally with:

- Lightautoml (LAMA) - I also share some public notebooks below.
- Xlearn FFM
- ResNet/RealMLP/DCN-V/DNET - various deep tabular algorithms - for this I used LLMs to write the code and run locally on my machine.
- Lots of feature engineering variations for all of these models.
- Tab Transformer - my public kernel below.

#### AutoML

I tried several variations of AutoML:

* Autogluon (different presets + feature engineering) - see for a few notebooks below
* LightautoML (LAMA) - same as autogluon + feature engineering + changing presets/fitted models

Overall I underused these last few months and they definitely added a lot of diversity to my ensemble.

#### Public Kernels

I used a lot of Kaggle notebooks this month in a couple of ways:

- Running notebooks that my GPU (a mobile RTX 4060) can't run - see my notebooks below.
- Using others' feature engineering with my models and others' models with my features.

I publish some of my models as notebooks and reference them below.

#### Ensembling

I tried:

- Hill climbing - thanks to @cdeotte for https://www.kaggle.com/code/cdeotte/gnn-starter-cv-0-9155-with-hill-climbing-demo
- `LinearRegression` - both rank-based and not rank-based
- `Ridge` with various alphas

In the end a simple `LinearRegression` was the best in both Public and Private LB score.

The recipe given `10` submissions per day was simple:

> Collect OOF/test predictions from models 
> Add **one model at a time** and observe CV-LB scores.
> Be careful when CV scores go up too much but LB scores stay the same and vice-versa.

In the end hill-climbing chose too few models and wasn't as good as `LinearRegression`.

[final ensemble]

#### Miscellaneous

Due to time constrains this month I did a few things that I wouldn't recommend.
I'll list them here for full transparency:

- I used models with different # of folds and seeds - due to my simple ensembling (`LinearRegression`) at the end this didn't seem to be much of an issue, but it _can absolutely be_. I do believe it lead to my CV-LB gap being bigger (see https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679436).
- I used ensembles as models for `LinearRegression`. This isn't necessarily "bad" and I think it helped because the small 1-2 model ensembles optimize for ROC-AUC directly, whilst `LinearRegression` doesn't.

#### Thank You

A big thank you to all Kagglers below whose kernels I used as a starting point, for features, or for models.

Special thanks to @yekenot for RealMLP ( https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit ) as well as @yunsuxiaozi for RealMLP from scratch (https://www.kaggle.com/code/yunsuxiaozi/realmlp-from-scratchcv-0-91908 ).


With a few edits (I publish my edits here - https://www.kaggle.com/code/include4eto/ps-s6-e3-realmlp-pytabkit-drop-non-te-cats) and here (
https://www.kaggle.com/code/include4eto/realmlp-from-scratch-feature-engineering) these were some of the best single models I used this month!

### References

#### External Kaggle Notebooks (by author)

**@cdeotte (Chris Deotte)**
- [chatgpt-vibe-coding-3xgpu-models-cv-0-9178](https://www.kaggle.com/code/cdeotte/chatgpt-vibe-coding-3xgpu-models-cv-0-9178)
- [gnn-starter-cv-0-9155-with-hill-climbing-demo](https://www.kaggle.com/code/cdeotte/gnn-starter-cv-0-9155-with-hill-climbing-demo)

**@lightningv08**
- [s6e3-cv-0-91849-xgb-kfold-fe-pl](https://www.kaggle.com/code/lightningv08/s6e3-cv-0-91849-xgb-kfold-fe-pl)
- [ps-s6-e3-realmlp-pytabkit](https://www.kaggle.com/code/lightningv08/ps-s6-e3-realmlp-pytabkit)
- [ps-s6-e3-trompt-pytorch-frame](https://www.kaggle.com/code/lightningv08/ps-s6-e3-trompt-pytorch-frame)

**@badalkrsharma (Badal)**
- [xgb-lgb-cv-0-91631](https://www.kaggle.com/code/badalkrsharma/xgb-lgb-cv-0-91631)

**@mikhailnaumov (Mikhail Naumov)**
- [customer-churn-ensemble](https://www.kaggle.com/code/mikhailnaumov/customer-churn-ensemble)

**@blamer / @blamerx**
- [auc-0-91925-xgboost-bi-tri-target-encoding](https://www.kaggle.com/code/blamerx/auc-0-91925-xgboost-bi-tri-target-encoding)
- [s6e3-0-91902-optimized-catboost](https://www.kaggle.com/code/blamerx/s6e3-0-91902-optimized-catboost)
- [s6e3-ridge-xgb-n-gram-0-91927-cv](https://www.kaggle.com/code/blamerx/s6e3-ridge-xgb-n-gram-0-91927-cv)

**@sdeograde**
- [customer-churn-cat-realmlp](https://www.kaggle.com/code/sdeograde/customer-churn-cat-realmlp)

**@dmahajanbe23**
- [customer-churn-histgradient-boosting-0-91367](https://www.kaggle.com/code/dmahajanbe23/customer-churn-histgradient-boosting-0-91367)

**@thomaswesthead**
- [vibe-coded-5xgpu-models-cv-0-9182](https://www.kaggle.com/code/thomaswesthead/vibe-coded-5xgpu-models-cv-0-9182)

**@liufengkai**
- [model-with-more-features](https://www.kaggle.com/code/liufengkai/model-with-more-features)

**@varadvaste**
- [realmlp-basic-auc-0-91406](https://www.kaggle.com/code/varadvaste/realmlp-basic-auc-0-91406)

**@yunsuxiaozi**
- [realmlp-from-scratch](https://www.kaggle.com/code/yunsuxiaozi/realmlp-from-scratchcv-0-91908)

**@ashishsinghrawat**
- [s6e3-single-realmlp](https://www.kaggle.com/code/ashishsinghrawat/s6e3-single-realmlp)

**@eric (fork of cdeotte)**
- Fork of cdeotte's vibe-coded 3xGPU models

#### My Kaggle Notebooks (@include4eto)

- See notebooks below. I left them as "copied from" to show attribution from previous competitions (and this one).

Update: Final notebook and dataset added - https://www.kaggle.com/code/include4eto/3rd-place-solution-linearregression-stack and https://www.kaggle.com/datasets/include4eto/ps6e3-all-oofs.
