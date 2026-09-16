# 25th place solution

Competition: commonlit-evaluate-student-summaries
Rank: #25
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446701

Congrats to all the winners!
Thanks to organizers and all participants! I really enjoyed this exciting competition!

This is my final solution!

# Overview

As [I published before](https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-feature-engineering), I used Deberta + LGBM tric.



## CV-LB Score Summary 

|  | wording CV | content CV | CV | Public | Private  |
| --- | --- | --- | --- | --- |--- |
| DebertaV3-base | 0.425 | 0.542 | 0.483 | 0.490 | 0.478 | 
| DebertaV3-large | 0.412 | 0.534 | 0.477 | not submit | not submit |
| DebertaV3-base + DebertaV3-large with LGBM tric | 0.419 | 0.551 | 0.485 | 0.433 | 0.463 |

## 1st stage: Deberta



- base model : DebertaV3-base, DebertaV3-large
- input: `prompt_text` + `[SEP]` + `prompt_question` + `[SEP]` + `text`
    - max_len = 2000
- loss: MCRMSE loss
- custom header: average pooling (using `text` only)
- inference: top 3 cv score model average (to avoid inference timeouts)

|  | wording CV | content CV | CV | Public | Private  |
| --- | --- | --- | --- | --- |--- |
| DebertaV3-base | 0.425 | 0.542 | 0.483 | 0.490 | 0.478 | 
| DebertaV3-large | 0.412 | 0.534 | 0.477 | not submit | not submit |

By the way, I show the score models trained using all data without cutting CV.

|  | wording CV | content CV | CV | Public | Private  |
| --- | --- | --- | --- | --- |--- |
| DebertaV3-base (full data, epoch=2)  | -  | - | - | 0.470 | 0.481 | 
| DebertaV3-large (full data, epoch=2) | - | - | - | 0.480 | 0.465 |

It seems that using single model (DebertaV3-large,full data train, epoch=2) can achieve silver medal score. 

### Why I used `prompt_text`?

I think many participants suffered from large `wording`  rmse.

In my opinion, this big error comes from  [score rotation](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/430705) by competition host, I think.



If the content and wording data we get from this competition has already been rotated 30 degree, the score we get from competition is like below.

- content score (apparent) = original_content score * cos30° + original_wording score * -sin30°
- wording score (apparent) = original_content score * sin30° + original_wording score * cos30°

Thus, if my idea is correct, the content and wording score dataset we get from competition will contain both original content and wording components.

The meanings of content and wording score are shared from host, and original content score is based on following components.

- Main Idea
    - How well did the summary capture the main idea of the source?
- Details
    - How accurately did the summary capture the details from the source?
- Cohesion
    - How  well did the summary transition from one idea to the next?

I think apparent wording score contains not only wording contents score but also original contents score. So, in order to improve the accuracy of the apparent wording, I think it necessary to accurately predict the original content score and use `prompt_text` for predicting original content score.

(No one don't know what the truth is. This is just my opinion...)

### Why I used custom header?

So far, I have shown that it is better to use prompt_text.

However, I think that simply using prompt_text will result in larger input tokens, which in turn will carry more noise. As a result, I think that we can’t get a good cv score.

So I used the custom header to input only the `text` portion into the final layer.
I think we can use `prompt_text` and `prompt_question`, but focus the deverte on the `text`.

## 2nd stage: LightGBM

 I use LGBM tric as I published.

### Using features

- deberta output 
  - content_debertav3_large_prompt_text_text_ver2_pred
  - content_debertav3_base_prompt_text_text_ver2_pred
  - wording_debertav3_large_prompt_text_text_ver2_pred
  - wording_debertav3_base_prompt_text_text_ver2_pred
- handcrafted features
  - num_chars
  - num_unique_words
  - embedding_sim (sentence transformer)
  - tfidf_sim
  - subjectivity
  - trigram_overlap_ratio
  - difficult_words
  - coleman_liau_score
  - polarity
  - nn_count
  - summary_length
  - dale_chall_readability_score
  - num_stopwords
  - mean_tokens_in_sentence
  - bigram_overlap_ratio
  - flesch_reading_ease
  - jj_count
  - max_count_per_sentence
  - vb_count
  - mean_count_per_sentence
  - osman
  - linsear_write_formula
  - gunning_fog
  - word_overlap_count
  - splling_err_num
  - szigriszt_pazos
  - smog_index
  - median_count_per_sentence
  - pr_count
  - num_words_title
  - min_count_per_sentence
  - num_punctuations
  - fernandez_huerta
  - gulpease_index
  - gutierrez_polini
  - crawford
  - automated_readability_index
  - bigram_overlap_count
  - trigram_overlap_count
  - flesch_kincaid_grade
  - num_sentences
  - cd_count
  - quotes_count
  - num_words_upper
  - num_paragraphs
  - uh_coun
