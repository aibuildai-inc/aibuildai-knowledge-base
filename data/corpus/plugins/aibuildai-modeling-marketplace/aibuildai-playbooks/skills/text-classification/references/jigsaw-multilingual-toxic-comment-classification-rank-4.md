# 4th place solution

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #4
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160980

Thanks to kaggle for hosting such an interesting and challenging competition! Multilingual NLP is something very interesting yet difficult. In hindsight I (Dieter) wish I’d spend less time on the tweet sentiment extraction competition and more on this one. 

# Brief Summary
Our solution is a simple blend of several transformer models (mainly xlm-roberta-large) paired with some post-processing. The architecture was the same as in public available kernels with the classification head taking either max+mean pooling of hidden states or the hidden state of the CLS token. We used a 3-step approach for training our models, where starting from 7 languages we fine-tuned to 3 languages and finished at a single language. As this 3-step approach results in distortion of global predictions we shifted languages individually by a factor as a post-processing step. 

# Detailed Summary
We are still astonished by the great result in such a short time. While @aerdem4 worked a bit longer on this competition and already had a good understanding of specific challenges @cpmpml and @christofhenkel joined right after the tweet sentiment competition a few hours before team merger deadline. Two days ago we were still at a 100+ spot and were hoping for a silver medal at best. But with persistence and the right amount of intuition what ideas to proceed with, and of course some luck we managed to climb right to the upper gold position.

## Preprocessing
None. That's NLP in 2020 :D

## Speeding up training
We did two things.  One was to downsample negative samples to get a more balanced dataset and a smaller dataset at the same time.  If you only use as many negative as positive then dataset is reduced 5x roughly, same for time to run one epoch.  Some of our models were trained that way.

Another speedup came from padding by batch.  The main idea is to limit the amount of padding to what is necessary for the current batch.  Inputs are padded to the length of the longest input in the batch rather than a fixed length, say 512.  This is now a well known technique, and it has been used in previous competitions.  It accelerates training significantly.  
We refined the idea by sorting the samples by their length so that the samples in a given batch have similar length.  This reduces even further the need for padding.  If all inputs in the batch have the same length then there is no padding at all.  

Given samples are sorted, we cannot shuffle them in training mode.  We rather shuffled batches. This yields an extra 2x speedup compared to the original padding by batch.  Training one epoch for xlm-roberta-large on the first train set takes about 17 minutes on a V100 GPU.

## Cross-validation
As time was short and hence we did not have many submissions to spare for getting LB feedback we worked on getting a reliable cross-validation. We used a mix of Group-3fold per language and simple 3fold of the validation data to represent the unknown languages ru,pt,fr as well as the known languages it, es and tr. which results in a 6fold scheme. So do for example fold1 represent what if es would not be in valid.csv which should behave the same as pt is not in valid. Taking the mean AUC of all folds was a very good proxy for Public and Private LB.

## Architectures
As most teams we used xlm-roberta large as the main backbone with a simple classification head that either uses max+mean pooling or the hidden states of the CLS token. In total we used the following backbones with their approximate weighting in our final ensemble in brackets.
 
- 5x xlm-roberta-large (85%)
- 1x xlm-roberta-base (5%)
- 1x mBart-large (10%)

## Training strategy
One main ingredient for our good result is a stepwise finetuning towards single language models for tr, it and es. Let us illustrate the whole approach before explaining:



As input data we used the english  jigsaw-toxic-comment-train.csv and its six translations available as public datasets which combined are roughly 1.4M comments. I realized that training a model directly on the combined dataset is not optimal, as each comment appears 7x in each epoch (although in different languages) and the model overfits on that. So I divided the combined dataset into 7 stratified parts, where each part contains a comment only once. For each fold we then finetuned a transformer in a 3step manner:


- Step 1: finetune to all 7 languages for 2 epochs
- Step 2: finetune only to the full valid.csv which only has tr, it and es
- Step 3: 3x finetune to each language in valid resulting in 3 models

We then use the step1 model for predicting ru, the step2 model for predicting pt and fr and the respective step3 models for tr, it and es. Using the step2 model for pt and fr gave a significant boost compared to using step1 model for those due. Most likely due to the language similarity between it, es, fr and pt.

We used mainly a batchsize of 32 using gradient accumulation and a learning rate of 3e-6 with linear decay with AdamW optimizer. One thing worth mentioning is that we train on a max sequence length of 512 due to the dataloader mentioned above which sorts the data by length and then randomly serves batches of same length comments. Only the huge speed-up paired with the non padding of shorter comments made using a 512 max length reasonable. 

Apart from one model which was built on top of a public kernel, all models were trained using pytorch and GPU.

## Ensembling
As for ensembling we did team member individual ensembling and then combined the models of the team members and public kernels by weighted sum of rank percentile 

## Post-processing
Another key ingredient, which we luckily found on the last day is post processing of the final predictions. Although languages are derived from individual models, the competition metric is sensitive to the global rank of all predictions and not language specific. So we took care of the languages having a correct relation to each other, by shifting the predictions of each language individually. In our final submission for example we used the following factors 

```
test_df.loc[test_df["lang"] == "es", "toxic"] *= 1.06
test_df.loc[test_df["lang"] == "fr", "toxic"] *= 1.04
test_df.loc[test_df["lang"] == "it", "toxic"] *= 0.97
test_df.loc[test_df["lang"] == "pt", "toxic"] *= 0.96
test_df.loc[test_df["lang"] == "tr", "toxic"] *= 0.98
```


We derived the factors by matching the test prediction mean with the public LB mean for each language individually. We had already made 6 submissions for probing the test set by setting one language to 1 and the rest to 0. Then we made sure that means are aligned to give similar experimental AUC. That posprocessing moved us from 14th place to 4th place on public LB! 

While this postprocessing might also help other teams, we think it specifically fixes the divergence of global prediction distributions introduced by having 5 models to predict 6 languages, and might not help much if a team used a single model approach.

Thanks for reading. Questions welcome.
