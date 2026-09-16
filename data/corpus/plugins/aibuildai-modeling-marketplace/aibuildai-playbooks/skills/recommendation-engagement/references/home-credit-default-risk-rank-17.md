# 17th place mini writeup

Competition: home-credit-default-risk
Rank: #17
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64503

I would like to thank fellow Kagglers for all discussions, ideas and kernels. This is my first competition and a wonderful one. I definitely learned a lot from it.

For my solution, I just want to share a bit about the feature engineering that gives me a significant boost:

I trained a model where training samples are each previous applications, and targets are the corresponding current targets, so I have a training set that looks like the following:

`SK_ID_CURR    SK_ID_PREV    AMT_ANNUITY     DAYS_DECISION ...     TARGET    prediction`

`1                            1               ....         ....               0        0.1   `
              
`1                            2              ....         ....               0        0.2`

`1                            3              ....         ....               0        0.1`

`2                           4              ....         ....               1        0.3`

`2                           5              ....         ....               1        0.4`

`3                           6              ....         ....               0        0.2`

`4                           7              ....         ....               1        0.1`

`4                           8              ....         ....               1        0.3`

`4                           9              ....         ....               1        0.5`

....

(assuming SK_ID_CURR = 1,3 have TARGET=0 and 2,4 have TARGET=1)

The features are those describing previous applications, so we can use the previous_application table directly, and for credit card, pos cash and installment tables, we do groupby('SK_ID_PREV') instead of groupby('SK_ID_CURR'). We don't need to worry about if it is optimal to average over previous applications of different time or different amount.
This way, the model will find the correlation between a specific previous application and current probability of defaulting, or "what is the probability of a certain previous application belongs to someone who has defaulted loan currently". After we get the prediction for each previous application, we can do:

agg_prev_score = df.groupby('SK_ID_CURR')['prediction'].agg({'mean','max','sum'...})

The aggregated predictions are pretty good features, and we can merge it to our regular training set by 'SK_ID_CURR'. So instead of doing aggregations like mean/sum on previous applications as people normally do, we look at each previous application separately and aggregate later, I guess this would give some complementary view on the dataset.

And we can do the same thing on each bureau record as well. These generated features boost my CV from .798 to ~ .801.

==============

Edit: Github [link](https://github.com/NoxMoon/home-credit-default-risk) to my solution.
