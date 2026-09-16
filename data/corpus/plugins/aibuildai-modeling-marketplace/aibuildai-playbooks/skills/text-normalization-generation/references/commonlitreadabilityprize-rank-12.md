# 12th (public 19th) place solution

Competition: commonlitreadabilityprize
Rank: #12
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/260325

First of all, Thanks Kaggle and the host for organizing this interesting competition.
I also want to thank the kagglers who made this competition so much fun for me. Thank you so much!

my final inference notebook: https://www.kaggle.com/kentaronakanishi/clrp-095-ensemble13-cv-kaggle93

# Summary

Final submissions are 13 and 11 models ensemble. The weights of ensembles are calculated by Nelder-mead with an objective using CV and LB.
 
| Name | CV | Public LB | Private LB |
| --- | --- | --- | --- |
| ensemble 13 | 0.4442 | 0.450 | 0.449 |
| ensemble 11 | 0.4451 | 0.449 | 0.449 |


I pre-trained huggingface base models by wikipedia (simple/normal) data with MLM and pseudo target, and finetuned them by competition data.
I didn't use other external data because I can't judge it is allowed to use scraped data.

Works with big improvements are re-initialization, pre-training with pseudo target, and ensemble.

# Models

Base models for 13 models ensemble are nine `microsoft/deberta-large`, two `albert-xxlarge-v2`, and one for `google/electra-large-discriminator` and `roberta-large`.

| Name | CV | Public LB | Private LB |
| --- | --- | --- | --- |
| `deberta-large` multi custom head | 0.474 | 0.464 | 0.468 |
| `deberta-large` multi custom head | 0.467 | 0.462 | 0.467 |
| `deberta-large` attn head | 0.467 | 0.461 | 0.463 |
| `deberta-large` multi custom head | 0.465 | 0.457 | 0.457 |
| `deberta-large` multi custom head | 0.463 | 0.456 | 0.456 |
| `roberta-large` multi custom head | 0.475 | 0.465 | 0.456 |
| `deberta-large` multi custom head | 0.462 | 0.456 | 0.460 |
| `deberta-large` multi custom head | 0.463 | 0.455 | 0.459 |
| `deberta-large` multi custom head | 0.463 | 0.455 | 0.460 |
| `deberta-large` with small (not pre-trained) network | 0.467 | 0.458 | 0.458 |
| `alberta-xxlarge-v2` cls head | 0.475 | 0.461 | 0.460 |
| `alberta-xxlarge-v2` multi custom head | 0.467 | 0.463 | 0.454 |
| `electra-large` multi custom head | 0.471 | 0.464 | 0.461 |


Best public LB by single model is 0.453, but this is not used for ensemble.

custom head is a combination of cls, avg pool, max pool, attn pool. Each head has output layer to target value, and then average (or weighted average) them.

# Training

## Basic settings

```yaml
num_folds: 5
num_epochs: 6
lr_scheduler_epochs: 10
learning_rate: 0.00002
learning_rate_output: 0.0001
batch_size: 16
max_length: 256
validation_interval: 10
```

## Pre-training

I have pre-trained again for huggingface pre-trained models with wikipedia data. I used data by @markwijkhuizen . Thanks a lot!

Not only MLM, I use pseudo labeling by best ensemble model on that time for wikipedia data and train with it. The details is below:

1. making good models and ensemble
2. predicting target for wikipedia data (simple/normal) by best ensemble models. This is predicted by each fold to prevent target leakage.
3. filtering wikipedia data by
    - has excerpt in both of simple and normal
    - the pseudo-target by pseudo labeling looks correct: satisfying `simple - margin > normal` where margin = 0.5
4. pre-training by wikipedia data with tasks: MLM and predict pseudo target.

This pre-trained and fine-tuning framework improved scores a lot in public LB.

## Others

- re-initialization top N layers of pre-trained model. N=5
    - worked a lot
    - [https://arxiv.org/abs/2006.05987](https://arxiv.org/abs/2006.05987)
- layerwise LR
- RDrop
    - maybe work a little
    - [https://arxiv.org/abs/2106.14448](https://arxiv.org/abs/2106.14448)
- dropout output layer 0.2 ~ 0.3
- LAMB optimizer make training stable (but no improvement on score)

# Not worked for me

- classification approach
    - cross entropy
    - rmse with softmax weighted sum by center value
    - histgram loss
        - same or little worse performance as regression approach
        - [https://arxiv.org/abs/1806.04613](https://arxiv.org/abs/1806.04613)
- pairwise comparison and simulate bradley terry approach
    - make pairs by combination of all excerpts and predict which is easier to read.
    - pairwise comparison is harder than regression for me
- linguistic feature
    - little or not worked with output layer or with features of svm
- hidden_dropout = 0.0
    - CV improves but LB didn't change or even worse
- GaussianLoss, MAE, Huber, RankLoss have no improvements
- Target balanced loss (the bigger abs target, the bigger loss) not improvement
- pseudo labeling noisy training data
    - CV off-course improved but LB not.
