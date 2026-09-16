# 8th place solution [LB 0.20138]

Competition: gendered-pronoun-resolution
Rank: #8
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90344#latest-523450

My approach was based purely on fine-tuning large Bert models and heavy ensembling to leverage the instability of fine-tuned Bert models. I started by fine-tuning large Bert on several tasks, most of them about pronoun resolution (swag,  definite pronouns, 2 flavours of PreCo and WinoBias). Each of those models is then fitted on the GAP dataset a number of times (10+). Afterwards, I ensemble, through simple averaging, the top 50% models by validation loss per task. At this point, I have 5 different set  of test predictions (ensemble_swag.csv, ensemble_wino_bias.csv, ensemble_preco.csv ...). Ensembling those, again through simple averaging we arrive at our final prediction set.

Several pre-training task (QA, classification, etc...) have been tried and most hurt results. Bert large vanilla directly tuned on GAP also hurt overall results when added to the ensembling. It seems that, with the exception of Swag, only directly related fine-tuning task have yielded positive results. 

Generally, the best hyperparameter for any fine-tuning has been: learning_rate = 1e-5, batch_size = 32, warmup_proportion = 0.1, epochs = 2. 

I believe the reason ensembling worked so well is the instability of the model. Different seed could yield an evaluation loss from 0.33 to 0.40, so it seemed likely that those models learnt different things. Ensembling brought the stage 1 test loss from .33 to .296.

I had issues getting the conll2012 dataset but I believe the coreference task would of ensembled well with the rest.

Congratulations to all and thank you for organising such a cool competition!
