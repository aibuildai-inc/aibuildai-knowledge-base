# 35th Place Solution

Competition: commonlit-evaluate-student-summaries
Rank: #35
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446591

Thanks for organizing the challenging competition. Though we got a silver medal finally, we learned a lot from the experienments. And thanks my teammates @penpentled @shigeria @chaudharypriyanshu @takanashihumbert, they have been really helpful these months. 

# 1. Summary

Our best private solution submitted is a ensemble of models, which contain a model with prompt_text, 3 wording models and a content+lgbm model without prompt_text. The inference time is about 8hrs. 
| Sub | Type | CV | LB | PB | Comment |
| --- | --- | --- | --- | --- | --- | 
| No.1 | best lb | 0.4655 | 0.426 | 0.466 | model with prompt_text training max_len=896 infer 1024 |
| No.2 | best cv | 0.463 | 0.427 | 0.467 | substitute the content model with a distilled model+lgbm |
| No.3 | conservative | 0.476 | 0.439 | 0.484 | only models without prompt_text |  


Though we have a model scored 0.461 for pb, we didn't select it since it only scored 0.445 for lb. It contains training on 2 folds validation on other 2 folds models by @takanashihumbert and the base model with prompt_text.

# 2. Training
Models submitted are shown below.
| ID | Type | Train max_len | Infer max_len | Text Cleaning | CV | Model | With prompt_text |
| --- | --- | --- | --- | --- | --- | --- |
| base | both | 896 | 1024 | public methods | 0.477 | v3-large | ✓ |
| w2 | wording | 512 | 512 | replaced copied text with [PASSAGE] | 0.556 | v3-large |
| w4 | wording | 768 | 768 | replaced copied text with [PASSAGE] | 0.559 | v3-large |
| w6 | wording | 512 | 512 | replaced copied text with [PASSAGE] and [REFERENCE] | 0.5561 | v3-base |
| c1 | content | 1024 | 1024 | public methods | 0.4445 | v3-large |
| distill | content | 1024 | 1024 | joined copied text with [PASSAGE] and [REFERENCE] tokens | 0.488 | v3-base |

# 3. inference
We used optuna and nelder-mead to optimize for ensemble, and features that public notebooks shared for lgbm. We strived for both better cv and lb when ensemble.

# 4. Not working for us
- Models except for v3
- Mean Pooling, Concat Pooling, ...

# 5. Acknowledgements
- @tsunotsuno idea of lgbm and feature engineering

# 6.Notebooks
Best PB submission: https://www.kaggle.com/code/snorfyang/commonlit-exp-e6-shigeria1/notebook
Text cleaning, ensemble methods, final submissions and other notebooks may be published later.
