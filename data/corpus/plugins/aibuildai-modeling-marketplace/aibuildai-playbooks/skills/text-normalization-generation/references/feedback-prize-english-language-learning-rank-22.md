# 22nd Place Solution

Competition: feedback-prize-english-language-learning
Rank: #22
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369584

First, I would like to thank the competition hosts for organizing this competition and also the participants.
As a novice regarding NLP competitions and Kaggle, everyone's codes and discussions have been very helpful to me.
Luckily, I was able to get my first medal.

### Overview
I did a weighted average ensemble of "models trained on Feedback3 data" + "models trained on a mixture of Feedback1 pseudo-labeled data and Feedback3 data" + "models trained on Rapids SVR using my finetuned model embedding" + "models trained on Rapids SVR using huggingface pre-trained model embedding."

The CV strategy used MultiLabelStratifiedKFold with K = 5.
Incidentally, my use of this MultiLabelStratifiedKFold was buggy according to Discussion (see: https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/368437), but the correlation between CV and LB was good.

- [My inference code](https://www.kaggle.com/code/moritake04/private22nd-final-sub)

### What Worked
Several steps were taken to improve the model and will be explained.

##### Step 1. Train a good single model [Medium Impact]
First I tried to develop a good single model. After tuning by hand, the following configuration worked well.

- Epoch:
  - base model: 4
  - large, xlarge model: 2
- Learning Rate:
  - base model: encoder -> 1e-5, decoder -> 1e-4
  - large, xlarge model: encoder -> 1e-5, decoder -> 1e-5
  - Using Layerwise LR Decay
- Batch Size: 8
- Initialize the last layer
- Optimizer: AdamW (weight_decay = 1e-2)
- Scheduler: cosine_schedule_with_warmup (warmup_ratio = 0.1, num_cycles = 0.5)
- Criterion: SmoothL1Loss
- Head: Mean Pooling
- When using the xlarge model, freeze half of the layers
- gradient_clip_val = 1.0
- token length = 512 (and 4096 for ensemble, 512 was better)

Many people have achieved Public 0.43 with a single model, and that was my goal, but I just couldn't achieve it, so I decided to rely on an ensemble.

The highest Local CV at this step was 0.4502 for Deberta-xlarge (it was not Deberta-v3 series)

##### Step 2. Pseudo Labeling [High Impact]
I trained the following models in the above configuration and Pseudo Labeled the train and test data for feedback prize1 with a weighted average ensemble of these models.

- deberta-v3-base (token_length = 512)
- deberta-v3-base (token_length = 4096)
- deberta-v3-large (token_length = 512)
- deberta-v3-large (token_length = 4096)
- deberta-large (token_length = 512)
- deberta-xlarge (token_length = 512)
- muppet-large (token_length = 512)

I was careful to avoid leaks and labeled each of the 5 folds to generate a pseudo label. I also removed duplicates in feedback 1 data and feedback 3 data. It seems that preventing this leak was quite important in this competition.

Using the pretrained-model developed in step 1, I mixed the data from feedback 1 with the data from feedback 3 and fine-tuned it just for one epoch.

The highest Local CV at this step was 0.4472 for Deberta-xlarge. Here, the Public score for the single model reached 0.43, and furthermore, the Public Score for the ensemble model entered the bronze medal zone.

##### Step 3. Rapids SVR [Medium Impact]
First, I applied Rapids SVR to the embedding of the model developed in step 2, and although the CV was not good, both CV and LB increased when ensemble with the previous models, so I adopted it.

Next, I applied the model to a pretrained model available on huggingface. This model was also adopted since both CV and LB increased when ensemble with the previous model. At this stage, the ensemble model was able to enter the Public silver medal zone.

##### Step 4. Ensemble [High Impact]
The ensemble is a weighted average. The weights are optimized using the Nelder-Mead method. Since there were six targets in this case, I adjusted the weights for each target.

I also used the same weights for all 5 folds to prevent overfit to the local CV.

The final ensemble was the following model.

| Model | Local CV Score |
| ---- | ---- |
| deberta_v3_base_4096 | 0.4536 |
| deberta_large_512 | 0.4513 |
| muppet_large_512 | 0.4571 |
| deberta_xlarge_512 | 0.4502 |
| deberta_v3_base_512_pseudo | 0.4489 |
| deberta_v3_large_4096_pseudo | 0.448 |
| deberta_xlarge_512_pseudo | 0.4472 |
| deberta_v3_base_512_pseudo_svr | 0.4529 |
| deberta_v3_large_4096_pseudo_svr | 0.4531 |
| deberta_xlarge_512_pseudo_svr | 0.4538 |
| deberta_v3_large_4096_svr | 0.4519 |
| deberta_large_512_svr | 0.4553 |
| deberta_v3_base_4096_svr| 0.4542 |
| ensemble | 0.4436 |

In fact, I trained many more models, but these models were chosen as a result of manual and automatic selections.

Finally, I did an all data train using the procedure introduced here and took the average of the predictions of the 5 fold model and the all data train model as the final sub. I was able to choose the last sub without hesitation because Local CV and LB were quite correlated and Best Local CV = Best Public LB.

### What Didn’t Worked
- MLM
- AWP
  - I think it is my problem because some people are doing well.
- Initialization of two or more layers
- Stacking using predictions or embedded + text metadata (number of misspellings, number of words, number of punctuation marks, etc.)
- Add newline tokens
- Heads such as Weighted Layer Pooling, Attention Pooling, 1dcnn, lstm

### Important Citations
- https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train
- https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference
- https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x

### Thanks and Acknowledgements
Again, thank you to the organizers and participants of this competition, and I will continue to work hard to become a kaggle expert and a master.
