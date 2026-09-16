# 11th place solution

Competition: commonlit-evaluate-student-summaries
Rank: #11
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446776

Thank you Learning Agency Lab and Kaggle for hosting this competition. We are relieved 😌 to survive the shake up. 

**Context:** www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/description

**Data:** https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data

### Overview of the approach

Our selected submission had 17 diverse models. We used stacking with LGBM.  

We used one prompt per fold cv strategy. All our models had collate function which speeds up training as well as inference. Lower batch sizes were helpful as well. Most models had evaluation at the end of epoch, few models had evaluation multiple times during epoch. Many of the diverse and smaller models were distilled (described below). 

All of us were able to contribute models to the stack, some of us never checked individual models scores because public LB seemed unstable. Our stack seemed more stable than individual models. 

Apart from model predictions we had following features in stack: 
```python

word_overlap_count
bigram_overlap_count 
bigram_overlap_ratio 
trigram_overlap_count
trigram_overlap_ratio
        
- Text stat features
Osman
Gutierrez_polini
Dale_chall_readability_score
flesch_kincaid_grade

```

### Model Description

Here we describe each of the models : 


**(A)** **Deberta v3 large - full prompt text psuedo labeled model**  [ar3]
This model had best cv (0.4721) in stack - This was trained with max len 1790, attention head on top. This was also using back translated pseudo labeled dataset. Every sample was translated to german and then back to english. We randomly selected 50% of this data for training. So every training fold had different ~3100 samples that were pseudo labeled , validation set had no psuedo labeled samples.  
Loss function for this model was combination of SmoothL1Loss, RMSE Loss and RankLoss - @fightingmuscle described the loss function [here](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446554)
Input for this model - `[ANSWER_START] + summary_text + [ANSWER_END] + [PROMPT_START] + prompt_text [PROMPT_END] + [QUESTION_START] + prompt_question + [QUESTION_END]`
Inference maxlen = 1024

---
**(B)** 

- Deberta V3 Large [MK 43]  - MeanPooling
- Deberta V3 Large [MK 45]  - MeanPooling


These were the same models with different seeds and mean pooled, input to these were summary_text + prompt_question, RMSE loss, validation end of epoch. 


----

**(C)** 

-  Deberta V3 Large [r3] - CLS 
-  Deberta V3 Large [r4] - CLS
-  Deberta V3 Large [r5] - CLS 


These were the same models with different seeds and with CLS token, input to these were summary_text + prompt_question, MSE Loss,validation end of epoch. I added extra text “Summarize:” to prompt question , if we look 👀 at the data 🔎carefully, notice that there are prompts with "cite evidence" and these have much longer summary texts and it doesn't say summarize. After reading other solutions, should have tried some more variations to this. 

----

**(D)** 

- Bart Large  -  Attention Head (Distilled)
- Electra Large - Attention Head (Distilled)
- Funnel Large - Mean pooling (Distilled)

All these models were distilled - 50% oof labels and 50% true labels, with AWP added - only summary_text as input, validated multiple times during epoch. 

----

**(E)**
 
- Deberta v3 large [ar2] -  CLS Token  
- Deberta v3 base [ar4] -  CLS Token - Distilled model  
- Deberta v3 XSmall [ar5]  -  CLS Token - Distilled model   
- Deberta v3 XSmall [ar7] -   Distilled model - Bigram Signal 

---
Deberta v3 large [ar2] - Order of input to this model was different than the remaining models - `text + SEP + prompt_question` 

Deberta v3 Base and XSmall models [ar4, ar5, ar7] - These were distilled models. We utilized oof labels from a previous stack. These models were trained using oof labels where as validation set remained as true labels. 

Bigram Signal - 

```
# Thanks to @aman1391 and Google Bard for this one
outputs = self.model(input_ids, attention_mask, token_type_ids)
bigram_signal = outputs.last_hidden_state[:, 1:, :] * outputs.last_hidden_state[:, :-1, :]
feature = bigram_signal[:, 0, :]
return feature
```

**(F)**  **Deberta v3 large - Classification - Attention Head , Mean Pooling - (4x)** [ar6]
After analyzing data, we noticed there were 1134 distinct categories of content and wording, so we added a classification model with `BCEWithLogitsLoss`.  We trained 4 different models with different heads, averaged them together and then added to the stack. 



#### Other
We haven't understood how or why the splits for training/public lb/private lb were created. The distribution of prompts for training and public lb is very different than private lb. Very long prompt text in public and training set but not in private lb. Also number of summary text per prompt is very different for training/public LB and private LB, unfortunately these resulted in shake down for many many teams. 


*Many other things tried but were not part of selected submission*
- MLM based on this dataset shared  [here](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/441202) 
- RAG (Similar to what was shared in LLM )
- Many different architectures , heads, hyperparameters, loss functions 
- 2x T4 Inference 
- Large Distilled models 
- Adding more features to LGBM
- Catboost, XGBoost, NN alongwith LGBM
- Generating data from GPT3.5
- I'm glad we didn't spend much time on sliding window approach because the maxlen of prompts in private lb is less than or equal to public lb. 



**Acknowledgements**
Thank you @kononenko  for [this](https://www.kaggle.com/code/kononenko/pip-install-nlp-mit?scriptVersionId=142526877) compliant textstat kernel 

Thank you @tsunotsuno for [LGBM features](https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-spell-autocorrect) 

--- 

🙏 Thank you to my amazing teammates, grateful to have worked together on this with you all. 
- @aman1391 
- @phoenix9032 
- @ragnar123 
- @fightingmuscle
