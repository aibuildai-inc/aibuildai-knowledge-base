# 43th place solution (711 shake up)

Competition: commonlit-evaluate-student-summaries
Rank: #43
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446644

Thank you for organizers hosting the competition. I also appreciate everyone at Kaggle sharing insights through codes and discussions.

The reason why I was at very low rank (754 th) is that I didn't have enough time to improve public leaderboard score due to severe fever from 10/3 to the competition end.
(I have planed to reach similar CV score reported in [Single Model CV-LB discussion](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/424330) first, then try to improve CV and public leaderboard score. But there wasn't enough time to do later.)

# Overview of the approach

My solution is mean ensemble of two deberta-v3-large with different settings. Each model has 4 weights (4 folds), so the total number of weights for inference is 8. I have used best CV checkpoints.
CV setup is Group K fold by prompt_id and CV scores of each model are 0.482 and 0.492.

# Details of submission

Most influential part for CV score improvement is pooling inside student summary. Other things bring small improvements compared with that. Details are written in the following sections.

## Model input

Each model has different model input.
One is `[START] text [END] prompt_question [SEP] prompt_text`.
Another is `[START_QUESTION] prompt_question [END_QUESTION] [START_SUMMARY] text [END_SUMMARY] [SEP] [START_TITLE] prompt_title [END_TITLE] [START_PROMPT] prompt_text [END_PROMPT]`.


## Model architecture

Each model has different pooling.
One is mean pooling inside `[START]` and `[END]` tokens which means pool inside student summary.
Another is GeM pooling inside `[CLS]` `[SEP]` tokens which means pool inside prompt question and student summary.

Second model has separate heads to predict content and wording scores.

## Other settings

### Same settings across models

- Gradient clipping = 10.
- Set `hidden_dropout_prob` and `attention_probs_dropout_prob` are 0.
- Tokenizer max length is 1024.

### Different settings across models

- The frequency of CV computation to save best checkpoints during training is 100 and 300.
- Second model has different learning rate for heads.

# Things not worked

I have guessed most important thing of this competition is combining student summary and prompt text effectively. Most of time I have focused this direction but none of them bring significant improvements (They bring similar CV score of final models but even ensemble of them didn't give huge improvements).

- After passing summary and prompt_text to backbone, pool summary and prompt_text then take bert cross attention of them
    - I have also tried to stack bert cross attention and bert layer multiple times like [this paper](https://arxiv.org/abs/2112.03857), but not worked
- Pass summary and prompt text separately to the backbone, then concatenate those hidden states and pass to the head
- re initialization of layers
- freezing layers
- ensemble with other backbones (debrta-large, longformer)
- pretty long max length (4096)
