# 12th place solution

Competition: feedback-prize-effectiveness
Rank: #12
Source: https://www.kaggle.com/c/feedback-prize-effectiveness/discussion/347490

I would like to thank all the participants and hosts of the competition.
And thank you to the teams for working hard together ( @tomoyayanagi, @kashiwaba,  @zacchaeus, @shairahama )
Without team I would not be in this position.

# Model
- Deberta-large
- Denerta-xlarge
- Deberta-v3-large
- bigbird-roberta-large

## Token Classification
All models were built with token classification.
Token classification added a [CLS] token at the beginning of the discourse_text and a [SEP] token at the end.
The method of pooling was different in members, cls token, average pooling was used
The best input was to introduce special tokens (start, end) at the beginning and end (introduced by  @zacchaeus)

```
<{discourse_type} start> discourse text <{discourse_type} end>
```

## Adding meta description (Prompt)
We found a trend of labels for each essay topic. 
Based on these results, we added topic text to the top of the input.
This method gives us the best single model (build by  @zacchaeus )

Topic information was taken from the public notebook. Thank you very much.
(https://www.kaggle.com/code/jdoesv/take2-feedback-essays-to-prompts)

```
topic prompt [SEP] <{discourse_type} start> discourse text <{discourse_type} end>, ...
```
I think there was a discussion that the host posted that the topic of the essay is common. Therefore, I think it worked.

# Pseudo Labeling
At first, we used all of the 2021 data as pseudo-labels, but the training did not proceed well, probably because the distribution of the data was too different. Therefore, we adopted the method of sampling and using a portion of the data.

so, the following methods are good for us (This was introduced by @kashiwaba )

- Discard essays with high Adequate predictions in all the discourse_texts that make up the essay
- Get everything that contains Ineffective's predictions
- Sampling the rest of the essay including Effective at 25~35%

Pseudo labels matched folds trained base model to minimize leakage

Another consideration was to sample the 2021 data according to topic, since the distribution of topic is different in 2022, but there was not enough time

# Stacking
We have built a 2nd model with flattened 1st model predictions of the 3 labels as features.
Discourse_type is added as a meta-feature.
The model used logistic regression, lgbm and xgb.
Boosted cv and lb significantly more than weighted average using Nelder-Mead based ensemble

# What else worked or tips
- MLM
 - Smoothed loss and boosted cv/lb
- AWP
    - It seems to be good to start at epoch 1 or 2
- FGM
- Evaluate more validation in steps instead of epochs
- layer wise learning rate (decay rate 0.98)
 - Higher learning rate at transformer classification head
