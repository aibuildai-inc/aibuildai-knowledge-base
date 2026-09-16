# 11th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #11
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369409

# 11th Place Solution

First of all, I'd like to thank Kaggle and host for hosting this competition.
I still can't believe I got a solo gold.

## Overview


I ensembled 21 models (including models trained with Pseudo Labels). I used the Ridge-Regression and [Netflix method](https://kaggler.readthedocs.io/en/latest/_modules/kaggler/ensemble/linear.html#netflix) for ensembling.

For all of my transformer models I used the [this notebook](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train) and [this notebook](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference/notebook).


## Code
* Inference Notebook: https://www.kaggle.com/code/irrohas/final-many-deberta-ridge
* Code (Training): I will upload later.

## What Worked
* [Hight Impact] Ridge-Regression and Netflix method per target for ensembling
    * Result: about -0.009 
    * Reasoning and context: Adding the predictions of various models (even if the each score was low) to the ensemble, not just Deberta-v3, improved the subsequent CV. So I used 10 other models besides deberta-v3. 

* [Low Impact] Pseudo Label
    * Result: about -0.003
    * Reasoning and context: I created a pseudo label with the 'Feedback Prize - Evaluating Student Writing'(FB1) competition data. However, there were too many pseudo-labels for the training data, and using all of them was likely to result in a large leakage, so we limited the pseudo-data that could be used in each fold.
    I made up to 2 steps of learning with pseudo labels.

## What Didn't Work
* Stacking for 2nd stage prediction
    * What you did: I tried LGBM, 2DCNN, MLP, and GCN for 2nd stage prediciton.
    * Result: worse
    * Reasoning and context: I tried LGBM, 2DCNN, MLP, and GCN for 2nd stage prediciton. However, all of them made CV worse. Also the weight optimization by Nelder-Mead method did score better, but not as well as Ridge-Regression or Netflix-method, so we did not use it for final submission.
* Pretraiend from previous competition dataset
    * what you did: Before training, I pretrained model with data from a previous competition ('Feedback Prize - Evaluating Student Writing' and 'Feedback Prize - Predicting Effective Arguments')
    * Result: not change or worse
    * Reasoning and context: Scores were almost the same or slightly worse, so I stopped considering the cost of learning.
* Changing the model depending on the length of the input text
    * Result: worse
    * Reasoning and context: Deberta-v3 models can handle sentences of arbitrary length, while Roberta and Funnel were limited because of 512 and 1024. However, it was determined that even if the `max_len` limit was used, the model score would remain almost the same and could be used to seed the ensemble.

## Additional Context
### CV Strategy
I used [abiheshark's cv strategy](https://www.kaggle.com/code/abhishek/multi-label-stratified-folds) with 4,5 folds various seed.

### Training Config
* Oprimizer: AdamW
    * weight decay: 0.01
    * initial lr: 2e-5
* Loss: SmoothL1Loss
* No dropout


### My Score
My each model score is here, (Including models not in use)
I use mainly mean Pooling for model's head. (I tried LSTMhead and Attentionhead, but the score not changed)

|  model  |  CV  |
| ---- | ---- |
|  Deberta-v3-base |  0.4590  |
|  Deberta-v3-large  |  0.4519  |
|  Roberta-large  |  0.4583  |
|  luke-large  |  0.4590 | 
| bigbird-roberta-large | 0.4643 |
|  Deberta-large  |  0.4615  |
|  Deberta-large-mnli | 0.4629 |
|  Funnel-large   | 0.4686  |
|  Funnel-large-base | 0.4644 | 
|  Bart-large     | 0.4606  |
|  GPT2-medium     | 0.4658  |
| GPT2-large | 0.4882 |
|  mpnet-base  |  0.4952  | 
| electra-large | 0.4731 |
|  Deberta-v3-large (1st PL) | 0.4473 |
|  Debreta-v3-large (2nd PL) | 0.4467 |


My submissions CV score and Best Score CV are here,
|  submission  |  CV  |  My Inner Private Score Rank |
| ---- | ---- | ---- | 
| 20 models + Ridge | 0.4417 | 2 |
| 21 models + (0.5Ridge + 0.5Netflix) | Ridge:0.4417, Netflix:0.4411 | 3 |
| 13 models + Ridge | 0.4421  | 10 |
| 16 models + Ridge | 0.4420 | 1 |


### Model Selection
I made a very large number of models, but had to choose a model due to dataset storage, Deberta-v3 had the highest score, so I chose about half of them, and the rest I chose by looking at the CV of the various models. So electra and mpnet are not in the final sub. But the best private scores were the models that included them :_(

We also considered the need for a variety of Deberya-v3 models, so we used model weights from [public Code](https://www.kaggle.com/code/kojimar/fb3-deberta-family-inference-9-28-updated). I really appreciate.

## Important Citations:
* Code:
    * https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train
    * https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference/notebook
    * https://www.kaggle.com/code/kojimar/fb3-deberta-family-inference-9-28-updated
    * https://www.kaggle.com/code/abhishek/multi-label-stratified-folds
* Others:
    * netflix method: https://kaggler.readthedocs.io/en/latest/_modules/kaggler/ensemble/linear.html#netflix

## Thanks and Acknowledgements:
Finally, I'd like to thank Kaggle, host for hosting such an iteresting competition and all who participated in lively discussions.
I hope my solution helps you in your future NLP competitions!
