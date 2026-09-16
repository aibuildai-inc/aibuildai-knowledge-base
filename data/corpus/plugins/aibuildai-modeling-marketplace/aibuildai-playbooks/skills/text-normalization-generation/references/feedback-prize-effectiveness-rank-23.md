# Private 23rd solution

Competition: feedback-prize-effectiveness
Rank: #23
Source: https://www.kaggle.com/c/feedback-prize-effectiveness/discussion/347356

First of all, I would like to thank the competition organizers for this competition. Also, I would like to say thank you for all the people for providing great notebooks and discussions, especially @nbroad. As a matter of fact, my best single model is heavily based on [his kernel](https://www.kaggle.com/code/nbroad/token-classification-approach-fpe).

# stage1: bert prediction
Because simple text classification approach is not very good at score and took too much inference time, we abandoned this approach and adopt a token classification approach. We trained models different ways in the point of post process, averaging the prediction of all tokens in the discourse_text like US PPPM 8th solution or just adopting the prediction of the first sep token of the discourse_text. In the former type of models, we used weighted cross entropy loss and this helped a lot.

# stage2: lgb and xgb stacking and blending
Our best public sub adopt blending by Nelder-Mead and stacking using LightGBM and XGBoost by the ratio of 1: 1: 1. In stacking, some features such as the length of text, counts of each discourse type appearing each essay, and mean and std. of each discourse label in the essay_text. The details of some models used in ensemble is like below.
| model | token | CV | Public | Private |
| --- | --- | --- | --- | --- |
| deberta-v3-large | sep only | 0.5892 | 0.577 | 0.580 |
| deberta-v3-large | text token mean | 0.5907 | 0.579 | 0.587 |
| deberta-large | sep only | 0.5921 | Not submitted | Not submitted |


# summary
## useful attempt
-	Pseudo labeling using Feedback 2021 dataset
    -	    Both CV and Public LB decreased about 0.01.
-	AWP (eps: 1e-4, lr: 1.0, CV decreased about 0.003.)
-	LightGBM and XGBoost stacking and some feature engineering
-	segmented prediction and splicing like Feedback 2021 1st place solution
-	ensembling different models that used different inference ways (averaging predictions of all tokens in the texts or just using prediction of the first sep token of the texts)
