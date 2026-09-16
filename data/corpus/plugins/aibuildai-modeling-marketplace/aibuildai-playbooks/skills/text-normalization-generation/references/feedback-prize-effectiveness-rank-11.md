# [11th solution] Feature design and Knowledge distillation

Competition: feedback-prize-effectiveness
Rank: #11
Source: https://www.kaggle.com/c/feedback-prize-effectiveness/discussion/347386

First of all, thanks to the organizers to host this interesting competition and thanks to community for those great ideas which inspired me a lot. 

## Methods that work

- Pretraining
- Feature design
- Pseudo labeling
- Knowledge distillation

## Transformer Model

- Deberta-v3-large
- Deberta-large
- Deberta-xlarge

## Feature Design

1.Concatenate different `discourse_id` in same essay and predict token probabilities of tags:

> [CLS LEAD] lead discourse [END LEAD] [CLS POSITION] position discourse [END POSITION] ...

2.Predict token probabilities of `[CLS]` and `[END]` tags in essay directly:

> essay....  [CLS CLAIM] I think that the face is a natural landform because there is no life on Mars that we have descovered yet  [END CLAIM] essay ...

In my case, training model with whole essays works better than w/o essays like feature `1`, but it add up some diversity.

3.Add classifier tags:

> essay....  [CLS CLAIM] [Ineffective] [Adequate] [Effective] I think that the face is a natural landform because there is no life on Mars that we have descovered yet  [END CLAIM] essay ...

This time, the predicted probabilities are not `[CLS]` and `[END]`, but `[Ineffective]`, `[Adequate]` and `[Effective]`. And the activation function is using `Sigmoid` but not `Softmax`. Then the prob of these 3 tags would be scaling to 1. This could add diversity also.

 4.Predict all tokens in a span and take a mean value for discourse probabilities.

## Pretraining

I use [feedback-2021-data](https://www.kaggle.com/competitions/feedback-prize-2021/data) for model pretraining, it could improve model performance about ~0.003.

## Pseudo labeling

I spend a lot of time for pseudo labeling, especially when training model with pseudo labels (extra more data). 
In early stage I tried pseudo with hard label, but it did not give any improvement. And finally, soft label of probabilities give some  improvement in model performance (The loss function is using `soft cross entropy`).

## Knowledge Distillation (KD)

In my case, KD works better than pseudo labeling when model ensembling. I found that the higher of distillation loss weight,  the better cv but worse lb. So I take the high distillation loss in early stage of training, but lower weight in later stages, this could improve both cv and lb.

## Post process
I didn't have extra time to explore post process method, so I just used weighted average.

Anyway, solo in this game is a little bit hard. There left many things I have not tried yet.
