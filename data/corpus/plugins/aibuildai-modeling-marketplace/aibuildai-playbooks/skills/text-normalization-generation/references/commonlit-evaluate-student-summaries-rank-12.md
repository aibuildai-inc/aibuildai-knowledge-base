# 12th Place Solution

Competition: commonlit-evaluate-student-summaries
Rank: #12
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/447254

Thank you for organizing such an interesting competition!
I was unable to sleep due to the fear of shakedown in the last part of the competition, but I am happy to have remained in the gold medal position.

I would like to briefly summarize my methodology.

# Model & Result
My best private solution is an ensemble of long mode with prompt_text and short model without prompt_text.
The former is intended to evaluate that the text is summarized correctly, and latter to evaluate that sentence structure and word quality is good or bad.

## Long Model
</img>

Using prompt_text and long max_len worked very well for me. But inference time is long (over 6 hour) and I was not able to ensemble with other long model. So using the output of the Pooling Layer,  trained LightGBM to improve robustness.

## Short Model
</img>

Most of this model is based on [a very nice public notebook](https://www.kaggle.com/code/tsunotsuno/debertav3-lgbm-no-autocorrect).
The changes are as follows
* change deberta-base to deberta-v3-large
* non text cleaning
* drop prompt_length
* change definition of "overlap"
* freeze top 12 deberta layers  / non layers ensemble

## Ensemble Model
I simply ensemble (Long + Short)/2
</img>

# Not working for me
* Text Cleaning
* Backbone except deberta-v3-large
* MLM
* AWP
* Augmentation with ChatGPT
* SVR, xgboost (As an alternative to lightgbm)

Thanks.
