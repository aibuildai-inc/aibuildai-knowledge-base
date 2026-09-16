# 46th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #46
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369664

## Overview

Our solution was a large ensemble of around 30 models trained with various heads, optimizers, backbones, hyperparameters, MLM pretraining, and pseudo-label configurations. We also included a few SVR/Ridge models similar to [Chris's SVR solution](https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x).

We tuned six groups of ensemble weights, one per column. Doing this allowed us to train single-label models or models trained on a subset of the labels (i.e., we included a model trained to predict only `phraseology` and `vocab,` another only `syntax` and `conventions`), which helped with diversity.

We trained many additional models but kept only those that improved both CV and LB in the ensemble. However, in the final days of the competition, I worked on an ensemble that we optimized only for CV, which turned out to be our strongest submission.

## What Worked

### Pseudo-labeling

I used a subset of my submission ensemble to generate pseudo labels. The pseudo-labels generated had quite a different distribution from the training data labels. So, we selected only a subset of 2021 data that more closely matched the train data distribution. We included an extra 2x additional data with this approach.

We were careful to ensure the pseudo labels did not contain leaks by generating a set of pseudo labels per fold. We also excluded any text_ids that crossed over with this comp's data.

### Synthetic data

We trained a `t5-base` model to complete essays based on a prompt, including the first N words and the labels. We then used the model to create many synthetic examples, which we pseudo-labeled. We selected only the instances where the pseudo-labels closely matched the source label.

I doubled the original training dataset with these synthetic examples and added them in with the extra 2021 data.

### Many different pooling layers

* Mean Pool
* Mean Pool of concatenated hidden layers
* Concat Max/Mean/Min Pool
* Conv1d Pool
* Attention Pool
* Weighted Layer Pool
* Concat CLS token from multiple hidden layers.

### Many different backbones

deberta-v3-base was the best backbone, but the final solution included a deberta-v3-large, deberta-large, a deberta-xlarge, and a roberta-large.

I tried Bart and Funnel, but they didn't appear to work well on my CV.

### Single-label and subset-label  models

For any model that worked using all six labels at once, we also tried training a model with a single label. For a few examples, we tried 2 or 3 labels. Most of these models improved the ensemble CV, but only some improved the LB.

### Many different hyperparameters

- Different max length settings (512, 1024, 1256, and 1408)
- Various optimizers (Adam and Adafactor worked best)
- MLM pretraining (although this only worked for a few single-label models)
- Different epoch settings per label subset (e.g., more epochs for vocabulary single label model and fewer for syntax)

## What Didn't Work

- Vadim worked on many alternative pseudo-labeling approaches, including scraping essays and letters from the internet. I will get him to create a post describing that. Unfortunately, at this stage, Vadim is still without reliable power.

- Focusing on a strong single model, as is usually the best idea, seemed impossible. Training a model with the same configuration with only a differing seed gave wildly different LB and CV results. Any hyperparameter tuning appeared to affect the results randomly. Only pseudo-labeling consistently improved single model CV, and only ensembling reliably improved our LB score.

- We tried to use different Adversarial Training methods, such as the Fast Gradient Sign Method, Adversarial Weight Perturbation, etc. We did have some models that saw improved CV with AWP, but the increase in train time didn't seem worth it, especially with so many pseudo labels.

- Model stacking: We tried a few approaches but they didn't seem to give very strong CV results.

## Citations

- Huge thanks to this masterpiece of a kernel: [Utilising Transformer Representations Effictively](https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently) by @rhtsingh
- This kernel by @cdeotte: [RAPIDS SVR - CV 0.450 - LB 0.44x](https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x), which added good diversity to our ensemble.
- Many of the previous Feedback and other NLP competition solutions.

## Team

- @lexandstuff (me) from Brisbane, Australia
- @e0xextazy from Moscow, Russia
- @vad13irt from Dnipro, Ukraine
- @obatek from Tashkent, Uzbekistan

Unfortunately for us, in October, Vadim was left without reliable internet, power, and water after the Russian military began attacking civilian infrastructure in Dnipro and across Ukraine, leaving him unable to continue working and his family and their neighbors without basic human needs. Vadim's many important contributions in the first half of the competition gave us a solid foundation to continue in the 2nd half.
