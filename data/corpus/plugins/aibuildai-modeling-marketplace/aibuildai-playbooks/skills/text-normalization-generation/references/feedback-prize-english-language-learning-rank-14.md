# 14th place solution

Competition: feedback-prize-english-language-learning
Rank: #14
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369564

First of all, thanks to the host for an interesting competition and congratulations to all the winners!

# Overview
My final submission is ensemble of 21 models. The weights of the models are determined using nelder-mead. 

# Model
Using [this notebook](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train) as a reference, I experimented a lot, changing model and training method.

The following three things were important
- Add lstm after bert. And freeze the bert part for the first few epochs of training (model 15)
- Ensemble of low-cv models with negative weights(model 4, 12)
- Train model for each target or two targets.(model 20, 21)

| model | cv | weight|
| --- | --- |---|
|1.  bart-large |0.456|-0.009|
|2.  deberta-large |0.456 |-0.089|
|3.  deberta-xlarge|0.454|0.2175|
|4.  deberta-base |0.471|-0.148|
|5.  luke-large|0.456|0.072|
|6.  gpt2-large |0.460|0.109|
|7.  deberta-v2-xlarge | 0.456 |0.073|
|8.  deberta-v3-large (add lstm)| 0.452 |0.025|
|9.  gpt2|0.476|0.047|
|10.  deberta-v3-large(Re-initializing layer)|0.452|0.073|
|11.  deberta-v3-base (add lstm) | 0.453|0.014|
|12.  t5-large |0.462|-0.180|
|13.  gpt-neo-125M |0.479|0.052|
|14.  deberta-v3-large(pseudo label) |0.450|-0.013|
|15.  deberta-v3-large(add lstm + freeze bert(2epoch)) | 0.451|0.162|
|16.  deberta-v3-base(pseduo label + Re-initializing layer) | 0.454|-0.055|
|17.  deberta-v3-large(add /n token)|0.454|0.105|
|18.  deberta-v3-base(add /n token) |0.458 |0.032|
|19.  deberta-v3-base |0.456|0.109|
|20.  deberta-v3-large (train model for each target) |0.452|0.243|
|21.  roberta-large (train model for two targets) |0.458|0.159|
|ensemble |0.4435||
