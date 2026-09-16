# 19th place solution

Competition: commonlit-evaluate-student-summaries
Rank: #19
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446512

I'd like to start by expressing my gratitude to the organizers for hosting this competition and inspiring engaging discussions.
A special congratulations to the top participants for their outstanding achievements.
Personally, I was on the cusp of becoming a GM with a gold medal, and it's a bit disappointing to have come so close yet fall short. Nevertheless, your hard work and insights have truly made this a memorable event for everyone. 
Thank you all for making it such a rewarding experience.

And also thank you for my teammate takai380. 

## Overview

- A two-stage structure with the 1st stage being a Transformer and the 2nd stage being a GBDT (LightGBM).
- For the 2nd stage GBDT, a model is created for each out-of-fold (oof) of the Transformer, and finally, a Weighted Average is taken.
    - Some raw prediction values from the Transformers are also included in the final Weighted Average.
- Adjust the weight according to the total token count of summary_text and prompt_text.

[overview]

## Final Model

Three models with different features and weights were selected for the final submission:

- Sub1 (TrustLB)  LB:0.422 / CV:0.463-0.465 (Different CV calculation method)
- Sub2 (TrustCV)  LB:0.448 / CV:0.458
- Sub3 (Balance)  LB:0.441 / CV:0.460

## Transformer

- pao
  - model-p1: deberta-v3-large (question + summary_text)
  - model-p2: deberta-v2-xlarge (question + summary_text)
- takai
  - model-t1: deberta-v3-large (question + summary_text)
  - model-t2: deberta-v3-large (question + summary_text)
  - model-t3: deberta-v3-large (summary_text + prompt_text)
  - model-t4: deberta-v3-large (summary_text + question + prompt_text)

Predictions from model-t3 and t4 are also used in the final ensemble.
Model with prompt_text is strong, but it takes long time to inference, so just 2 models.

- For pao model:
  - Freezing layer
  - Concatenate the embedding of deberta-v3-large before training to the final layer (only for model-p1)
  - Prediction using CLS token
  - Linear scheduler with warmup, 3 epochs
- For takai model'
  - Freezing layer
  - concat pooling (concat multiple layer cls tokens)

## LightGBM

- Ultimately, about 80 features.
- 10 seeds x 4 folds of LightGBM Average for each Transformer model.
- To add features, check if the CV is improved with a score from a 50 seed average.


### Features
  - From public notebooks: (great thanks to @nogawanogawa)
    - https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-feature-engineering
  - Jaccard coefficient between question/prompt_text and summary_text.
  - Calculate tfidf for summary_text per prompt, and take the average tfidf per record.
    - For all of summary_text and only words not in prompt_text.
    - Taking average of all columns and average of non-zero elements.
  - Cosine similarity of average tfidf/BoW of summary_text per prompt.
  - Sentence Transformer embedding features.
    - Cosine similarity between prompt_text and summary_text.
    - Cosine similarity between question and summary_text.
    - Cosine similarity between average embedding of summary_text per prompt.
    - Calculate cosine similarity between embeddings of sentences split from prompt_text and summary_text, and take the standard deviation of similarity for all sentences.
    - Similarly, calculate cosine similarity between embeddings of sentences split from summary_text and prompt_text, and take the standard deviation of similarity for all sentences.
  - kNN-based features.
    - For each prompt, kNN with the embedding of summary_text.
    - Feature extraction of the average and cosine similarity of the top 5% of oof features with high similarity for each record.
    - The average of oof features also includes the difference from the average oof features per prompt.
  - Text similarity metrics:
    - BERT Score
    - ROUGE
    - BLEU
  - Learning Word2Vec with prompt_text for each prompt. Cosine similarity of the vectors of prompt_text and summary_text.
  - Include prediction values from the model of Feedback competition 3 as features.
    - https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference?scriptVersionId=104649006
  - Features from pyreadability.
  - Average the general frequency of words present only in summary_text and not in prompt_text for each summary_text.
- For bert_score and embedding features, the average embedding was taken from extracting text from the beginning, middle, and end for long prompts.

## Ensemble

- Weight optimization using Nelder-Mead.
  - Ultimately, only use those above 0 and round to a total of 1.0.
- Change ensemble weights according to the token count of summary_text + prompt_text.
  - For those with a certain number of tokens or more, gradually reduce the weight of raw Transformer predictions and increase the weight of LightGBM (linearly adjust the weight by token count).
  - It is assumed that when the token count is high, the prompt_text does not fit entirely in the Transformer with prompt_text input, reducing accuracy.
