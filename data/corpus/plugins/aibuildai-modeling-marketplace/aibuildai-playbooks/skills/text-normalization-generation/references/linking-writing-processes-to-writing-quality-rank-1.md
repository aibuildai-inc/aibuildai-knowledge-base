# [1st place solution] Data Cleaning+FE+External Data+Model Ensemble

Competition: linking-writing-processes-to-writing-quality
Rank: #1
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466873

[Note for latecomers] My rank was originally 2nd, but finalized to be 1st due to [some reasons](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/469199). I believe [the original 1st place solution](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/467154) is still worth reading.

# Thank You Everyone!
I'd like to thank the organizers and all participants. This is my first solo gold and monetary prize in four years of kaggle experience. I am so happy with it 🤩. 

The training and inference code is available [here](https://www.kaggle.com/code/tomooinubushi/2nd-place-solution-training-and-inference-code). This code includes so many redundant and/or unnecessary parts, but I opened it as is to avoid additional errors.

# Solution summary
[Summary of my solution]

# Data cleaning
In the era of deeplearning, data cleaning is one of the most under-emphasized parts of DS/ML. Since the training dataset of this competition is very small, I conducted data cleaning to reduce noises in train and unseen test dataset.

In summary, I conducted....
- Discard events ten minutes before first input.
- Correct up times from zero and consistently increasing (see [this discussion](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/447238)).
- Correct up times and down times so that gap times and action times are not too large (ten minutes and five minutes, respectively).
- Fix unicode errors in up events, down events, and text change columns with [ftfy](https://ftfy.readthedocs.io/en/latest/#) fix texts. [It reduced unseen up events (29 -> 24) and down events (30 -> 26)](https://www.kaggle.com/code/tomooinubushi/reduce-unseen-test-events-with-ftfy) revealed by @kononenko in [this notebook](https://www.kaggle.com/code/kononenko/lwp-unseen-test-activities-events)
- Discard events with Unidentified (see user ID 2f74828d)

# Sentence reconstruction and feature engineering
The first thing I did in this competition is to reconstruct sentences from keyboard activities. Since the scores are determined solely based on final text, I thought this is the most important point in this competition.
[This very early discussion topic](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/447735) by @kawaiicoderuwu and [this public notebook](https://www.kaggle.com/code/jasonheesanglee/updated-75-35-acc-revealing-hidden-words?scriptVersionId=148517101) by @jasonheesanglee were very helpful for me.
For better reconstruction, I improve some points.
- If cursor position and text change information do not match with reconstructed text, search sequence with nearest fuzzy match.
- Correct Undo (ctrl+Z) operation if cursor position and text change information do not match with reconstructed text

I did my best, but there are still 142 events with unexpected errors in training logs.

For feature engineering, [this public notebook](https://www.kaggle.com/code/hengzheng/link-writing-simple-lgbm-baseline) by @hengzheng was a good starting point. Learning from other notebook (especially this [public notebook](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features) by @awqatak was so impressive!), I used 378 features in total.
In summary, I used....
- Stats of Inter Key Latency, Press Latency, and Release Latency with gap 1
- Total counts of each activity and event
- Time to first reach 200, 300, 400, and 500 words
- Pause related features
- Word-time ratio, word-event ratio, etc.
- Stats of reconstructed text (e.g., words per sentence, word length etc.)
- Total counts of punctuation errors in reconstructed text (e.g., sequence like "qq qqq ,qqq" should be "qq qqq, qqq", and "qq qqq .qqq" should be "qq qqq. Qqq")
- Pause and revision burst related features.
- Tf-idf features of activities, events, and categorized Inter Key Latency.
- Word-level and char-level tf-idf features of reconstructed text
- Predictions of external essay scores based on word-level and char-level tf-idf features (explain later)

Once tf-idf features are extracted, I used truncated SVD to reduce their dimensions to 64.

Here is a (part of) feature importance plot of light GBM, though I never checked it throughout this competition 


# Use external data
As the training dataset is very small, I used external data about essay evaluation. I anonymized all essays and extracted tf-idf features with the same feature extractor with the training dataset. Then, I trained light GBM  models to predict external scores.  
The below scatter plot is the result of the model trained with [commonlit-evaluate-student-summaries](https://www.kaggle.com/c/commonlit-evaluate-student-summaries) data. 


To my surprise, the predicted external scores (x-axis) are well correlated with the scores of this competition (y-axis). Please note that light GBM models are trained only with anonymized external essays. I use these predicted external scores as features to increase generalizability of the models. I tried transformer models (e.g., deverta-v3), but they are never better than tf-idf+light GBM models.

I used eight datasets composed of 24 essay types.
* https://www.kaggle.com/c/commonlit-evaluate-student-summaries
* https://www.kaggle.com/competitions/asap-aes
* https://www.kaggle.com/competitions/asap-sas
* https://huggingface.co/datasets/whateverweird17/essay_grade_v1
* https://huggingface.co/datasets/whateverweird17/essay_grade_v2 (from [this discussion](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453372) by @thedrcat)
* https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview
* https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset
* https://www.kaggle.com/datasets/nbroad/persaude-corpus-2/

I also tried [keystroke dataset](https://www.cs.cmu.edu/~keystroke/), but never worked.

# Construct various models and ensemble them
Using both tree and NN models is important to improve the score. I learned light autoML in this [public notebook](https://www.kaggle.com/code/alexryzhkov/lgbm-and-nn-on-sentences) by @alexryzhkov. For classifier models, I used the framework shown in [this discussion](https://www.kaggle.com/code/alexryzhkov/lgbm-and-nn-on-sentences)

| Model |  CV| public LB |  private LB|
| --- | --- |--- | --- |
| LGBRegressor | 0.576  |0.576995 (late submission) |0.558459 (late submission)|
| LGBClassifier | 0.582 |- |-|
| XGBRegressor | 0.580 |- |-|
| XGBClassifier | 0.583 |- |-|
| CatBoostRegressor | 0.582 |- |-|
| BaggingRegressor | 0.594 |- |-|
| tabnet | 0.609 |- |-|
| light autoML densenet | 0.593 |- |-|
| light autoML resnet | 0.587 |- |-|
| light autoML fttransformer | 0.603 |- |-|


Ensemble
| Model |  CV|  public LB |  private LB|
| --- | --- |--- | --- |
| LinearRegressor | 0.572 |0.579468 (late submission) |0.557741 (late submission)|
| LogisticClassifier | 0.575 |- |-|
| Mean | 0.573 | 0.578796 |0.559289|


For ensembling, I also tried forward ensembling shown in [this public notebook](https://www.kaggle.com/code/cdeotte/forward-selection-oof-ensemble-0-942-private). Actually, forward ensembling gave me better public LB score of 0.575, but CV (0.575) and private LB score (0.560) were worse. I used nested CV of six bags x five folds stratified with score for final submission.

# Postprocessing
I just clipped predictions from 0.5 to 6.0. Rounding to 0.5, 1.0... never worked.

# Some remarks
Unlike [many people in the CV-LB thread](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/444947), my LB scores were always worse than CV scores. [As I have shown that score distribution of public LB dataset is similar to that of train set](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/456467), I always trust CV score in this competition.
Since I decided to use external data which impose extra inference time, I gave up efficiency prize. I selected three GPU models for submission. From these three models, the winning one is the model with the worst public LB score (0.578). Though I did my best for model generalizability, I have to admit that I am too lucky 😅. 

I learned a lot from many public notebooks and discussions. I listed them below, but there would be some missing ones.
* https://www.kaggle.com/code/hengzheng/link-writing-simple-lgbm-baseline
* https://www.kaggle.com/code/dangnguyen97/feature-eng-clean-outlier-lgbm-with-optuna#Train-OOF-LGBM-Models
* https://www.kaggle.com/code/ulrich07/tabpfn-and-xgboost-cv-0-19-lb-0-17
* https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features
* https://www.kaggle.com/code/hiarsl/feature-engineering-sentence-paragraph-features
* https://www.kaggle.com/code/jasonheesanglee/updated-75-35-acc-revealing-hidden-words
* https://www.kaggle.com/code/alexryzhkov/lgbm-and-nn-on-sentences
* https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs
* https://www.kaggle.com/code/cdeotte/forward-selection-oof-ensemble-0-942-private
* https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/447238
* https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/457385
* https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/444905
* https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/447735
* https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453372
