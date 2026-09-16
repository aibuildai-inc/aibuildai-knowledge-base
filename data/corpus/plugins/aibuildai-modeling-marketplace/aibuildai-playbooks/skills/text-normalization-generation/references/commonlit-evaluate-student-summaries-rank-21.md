# 21th place solution

Competition: commonlit-evaluate-student-summaries
Rank: #21
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/448625

First of all, I would like to thank the organizers for hosting such an interesting competition. Also thanks to kaggler for sharing so much useful information through notebooks and discussions.

I share my solution.

<br />
## Overview
My final submission is an ensemble of two deberta-v3-large models and LightGBM.
The pipeline is as follows.
.png?generation=1697805022893951&alt=media)

<br />


## Model
- Deberta v3 large
  - input_text = text+prompt_question+prompt_text
  - maxlen = 1024
  - CLS token
  - loss : RMSELoss
  - epoch = 4

<br />

- Deberta v3 base
  - input_text = prompt_title+len2text+prompt_question+text (※len2text : [[ref](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/437591)])
  - maxlen = 512
  - Attention Pooling
  - loss : RMSELoss
  - epoch = 4

<br />

- LightGBM
  - Almost the same as [publish nootbook](https://www.kaggle.com/code/tsunotsuno/debertav3-lgbm-no-autocorrect)


<br /><br />

## Kye Points
The following two points contributed significantly to improving the accuracy of the deberta model.
- Grouped-LLRD (Layer Wise Learning Rate Decay)
- freeze layer (freeze 4 layers close to embeddeding)  [[ref](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/433754#2405543)]


<br />

## CV
- validation strategy : GroupKFold(groups=prompt_id)

|  | 39c16e | 3b9047 | ebad26 | 814d6b | cv | public | private |
| --- | --- | --- | --- | --- | --- | --- | --- |
| model1 | 0.4768 | 0.5045 | 0.4461 | 0.5842 | 0.4948 | 0.466 | 0.471 |
| model2 | 0.4540 | 0.5154 | 0.4427 | 0.5830 | 0.4907 |   |   |
| LGBM | 0.4730 | 0.5802 | 0.4592 | 0.5809 | 0.5197 | 0.449 |0.481 | 


<br />

### Ensemble
- weighted average
  - cv : 0.4777
  - Public LB : 0.4351
  - Private LB : 0.46163

<br />


## Didn't Work
- text cleaning
- awp
- fgm
- svr 
- pseudo labels 
- etc…
