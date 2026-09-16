# 49th Place Solution [Single Model]

Competition: commonlit-evaluate-student-summaries
Rank: #49
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446516

Thanks a lot to the hosts and Kaggle for hosting this interesting competition. I personally had a great time working on this competition.

## Summary:
My solution primarily relies on a single DeBERTa-v3-large model with a custom MeanPooling layer. Used all the text columns `text`, `prompt_text`, `prompt_question`and `prompt_title` for training. Trainied with 1500 max_len and infer with 2048.  

## Input Processing:
To prepare the input data, I added two special tokens, '[SUMMARY_START]' and '[SUMMARY_END]', before and after the summary text, respectively. Then, I appended all the other prompt data with a '[SEP]' token. The final input text structure looked like this:
'[SUMMARY_START]text[SUMMARY_END][SEP]prompt_text[SEP]prompt_question[SEP]prompt_title[SEP]'.

## Training Details:
The DeBERTa-v3-large model was trained with a maximum sequence length of 1500 using the input structure mentioned above. I also modified the MeanPooling layer to calculate the mean only between the '[SUMMARY_START]' and '[SUMMARY_END]' tokens.

## CV - LB Details
* CV - 0.474 Public LB - 0.43 Private LB 0.469

## Inference Time:
It's important to note that this single model has a significant inference time of approximately 8.5 hours.

## What Worked: 
* Inferencing on larger Max Len
* Averaging Multiple checkpoints per epoch.

## What Didn't Work:

* Implementing a second-stage model (LGBM/XGBOOST).
* Text cleaning.
* Post-processing techniques, such as target smoothing.


## Training Code
https://github.com/rohitsingh02/CommonLit-ESS

## Inference Kernel 
https://www.kaggle.com/code/rohitsingh9990/commonlit-ensemble-new-v2?scriptVersionId=145818122
