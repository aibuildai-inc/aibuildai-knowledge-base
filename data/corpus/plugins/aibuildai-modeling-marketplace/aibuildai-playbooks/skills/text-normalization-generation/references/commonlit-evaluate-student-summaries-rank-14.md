# 14th place Solution (lucky for me)

Competition: commonlit-evaluate-student-summaries
Rank: #14
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446818

First my HUGE thanks to the competition host. what a fantastic competition! I indeeed had a fun time kaggling!
Also HUGE thanks to the shared notedbooks and the valuable discussions, from which I got a lot of inspiration and enlightment.

I'm not actually an NLP expert with all sorts of experiences and tuning skills, thus I felt very lucky that I could have got this far. Also because of the shake which fortunately turned out to be in my favour this time, doesn't really happen very often, :p.

Like many other top solutions, my model is acutally very simple deberta-v3-large. But the way I got there was rather serpentine.

### MY SOLUTION 

* **Inputs**:
Paried input into tokenizer: _summary text_ and _prompt title_ +[SEP]+ _prompt text_ +[SEP] + _prompt question_
```
prompt = sep.join([data["prompt_title"], data["prompt_text"], data["prompt_question"]])
tokenized = tokenizer(
        data['text'],
        prompt,
        padding=False,
        truncation=True, 
        max_length=config.max_seq_length,
    )
```

* **Model**:
deberta-v3-large. Like everybody else :P. I also tried some other models: deberta-v3-large-squad2, deberta-v3-base

* **Max Length**:
I tried different max lengths, the final model was with 1600.

* **Model set up and Hyperparameters**:
learning rate: after many rounds of hyperparameter tuning, I found 6e-6 the best
num_epochs: 4 
weight_decay: after hyperparameter tuning, 1e-4 
warmup_ratio: 0
batch_size: 2 for training and eval (I bascially cannot afford a larger batch_size due to GPU memory)
lr_scheduler_type: cosine
optimizer: adamw
I also tried freezing some layers, but did not boost local CV so the final model I actually did not freeze anything.\
No drop was used, no extra pooling was used.
Moreover, I found that local CVs and public LBs positively correlated with max length in training. Later the private LB proved this. Models with larger max lengths tend to have better private LBs.

* **CV Results and Ensemble**:
the best privat LB model achieved CV: 0.5513 (fold 814d6b) , 0.4587 (fold 39c16e), 0.5038 (fold 3b9047) and 0.4316 (fold ebad26)
The final model is a simple ensemble of the 4 folds. 
I actually considered to train the model using the set up above on the whole training data set. However, in the end I chose to do an ensemble of the models of different folds. This might not be the best practice, but gave me better opportunity to discover the relations between CV and public LB.
Public LB: 0.429
Private LB: 0.458

### Things did NOT work
1. **MLM did not work**
This was actually a major mistake I made in this competition. I took it for granted that MLM of deberta on previous commonlit datasets would help boosting the CV. With that in mind, I actually spent the whole August and the first half of September only training the models pretrained with MLM. The best model I got with MLM had however, local CVs of 0.5449, 0.4782, 0.5182, 0.4473 and public LB 0.445. This could not then be improved anymore.
It was not until the third week of September did I overthrow everything and started again without any pretraining using MLM and ended up with the model described above.

2. **Manually generated datasets with pseudo scores could help accelerate the training process but can lead to overfitting**
Almost at the end phase of the competition, I came up with the thoughts to expand the training data for pretraining task.  Here is how I did this:
  * Minor part of the extended training data: I downloaded from commonlit website some texts as prompt texts and questions as prompt questions. I then generated summary texts using gpt-4 by letting the machine mimick the way students write. The pseudo scores were given by the best model I created at that time. That gave me around 300 more training rows.
  * Major part of the extended training data: without any prompt text, I directly used the student written texts in the previous commonlit competitions and let my best model score them. That gave me ~4000 extra rows

  I hope this explains your concern @kononenko :p. 

I used the extended data to do pretraining task. I got better local CV results and earlier converged networks.  **HOWEVER**, the public LBs were not improving and in the end, the private LB turned out to be even worse :(. It could have led to overfitting (good lesson learnt). So my final model was not trained on this dataset.

3. **Lightgbm did not work**
I did not dive deeply enough to understand why lgbm did not work. I actually expected it could boost my CV.

### Things that I would like to try if I were given more time
1. EMA
I tried EMA in the early phase of this competition, with models pretrained by MLM. It did not help boosting the local CV so I didn't even bother submitting them. I would like to try EMA on the model without MLM pretraining if I had been given more time.
2. layer-wise discriminative learning rates 
same with EMA, i only applied this in my early models but did not boost local CV.
3. play with more max length set ups

### Here is a summary of the models I tried
*All LBs are the ensembles over 4 folds
| Model name|     Remarks      |  Local CV | LB | Private LB|
|----------|:-------------:|------:|
| Model 2308 |  pretrained with MLM, both pretraining and training are with max length 1536 | 0.5449, 0.4782, 0.5182, 0.4473 | 0.445 | 0.481|
| Model 2708 |  pretrained with MLM with max length 2048, model training with max length 1536| 0.5778, 0.4598, 0.5305, 0.4369 | 0.451| 0.466|
| Model 3008 |  pretrained with MLM, both pretraining and training are with max length 2048| 0.5532, 0.4706, 0.5226, 0.4383 | 0.447| 0.471|
| Model 1009 | pretrained with MLM. EMA used in training| 0.5631, 0.4728, 0.5201, 0.4407| not submitted| not submitted|
| Model 2609 (best model)| no pretraining, max length 1600| 0.5513, 0.4587, 0.5038, 0.4316| 0.429| 0.458|
| Model 0810| with incomplete extended dataset to pretrain, max length 1600| 0.5363, 0.4454, 0.4974, 0.4314|0.431 | 0.474|
| Model 1010| with complete extended data to pretrain, max length 1600| 0.5199, 0.4451, 0.4958, 0.4356| 0.458| 0.469|
|Model 0810 + lgbm| with complete extended data to pretrain, max length 1600, plus lgbm|  0.6154, 0.4695, 0.4990, 0.4391|0.435| 0.47|

As you can tell, with only a few exceptios, my public LBs and private LBs correlate somehow well.
