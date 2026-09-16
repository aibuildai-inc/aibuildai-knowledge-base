# 29th Place Solution for the CommonLit - Evaluate Student Summaries Competition

Competition: commonlit-evaluate-student-summaries
Rank: #29
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/447093

I would like to thank the host for organizing this competition and  all the participants who generously shared valuable information. I have learned a lot.


## Context
- Business context: https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview 
- Data context: https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data

## Overview of the approach
My approach is averaging the following models.  
- LightGBM stacking with meta-features of 9 deberta and bart models.  
- A deberta-v3-large model for the concatenation of summary text and prompt text. (Exp ID: 10)  

The key aspect of the latter model is an attention mask that focuses only on the summary text, as explained by other top solutions.

#### Model Training Settings

I trained the deberta models shown in this table.  
Out of these, 8 models are deberta-v3-large, and 2 model are bart-large.  
The bart-large models take prompt text as the encoder input and summary text as the decoder input. (However, I don't think they effectively utilize prompt text information...)  
CV strategy is Group k fold (group id = prompt_id)  

| exp id | model | input | pooling method | token max length | epoch | freezing layer | loss | awp | CV | Public LB | Private LB |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1 | deberta-v3-large | text | attention | 822 | 10 | 18 | smooth l1<sup>*</sup> | false | 0.542 | 0.494 | 0.541 |
| 2 | facebook/bart-large | text, question+prompt_text  | mean | 822 | 4 | - | smooth l1 | false | 0.555 | 0.514 | 0.556 |
| 3 | deberta-v3-large-squad2 | text | attention | 822 | 4 | 18 | smooth l1<sup>*</sup> | false | 0.581 | 0.496 | 0.549 | 
| 4 | deberta-v3-large | text | attention | 822 | 10 | 18 | smooth l1<sup>*</sup> | true | 0.535 | 0.488 | 0.532 | 
| 5 | bart-large | text+question, prompt_text | attention | 822 | 10 | - | smooth l1 | true | 0.533 | 0.486 | 0.510 |
| 6 | deberta-v3-large | question+title+text | attention | 822 | 10 | 18 | smooth l1 | false | 0.538 | 0.497 | 0.563 | 
| 7 | deberta-v3-large | question+title+text | attention<sup>**</sup> | 512 | 4 | - | smooth l1 | true | 0.628 | 0.519 | 0.535 | 
| 8 | deberta-v3-large | title+question+text | multi-scale<sup>***</sup> | 822 | 10 | 18 | smooth l1 | true | 0.535 | 0.477 | 0.538 | 
| 9 | deberta-v3-large | text | attention | 822 | 4 | 18 | mse + rank | false | 0.551 | 0.487 | 0.544 |
| 10 | deberta-v3-large | text+title+question+prompt_text | [See this section](#model-for-summary-text-and-prompt-text-input) | training : 1800, prediction : 1024 | 4 | 18 | smooth l1 | false | 0.504 | 0.461 | 0.47 |

Every training utilized layer-wise learning rate decay between 0.8 and 0.9.  
\*Sum of losses calculated in each of the last 3 hidden states.  
\*\*Two attention pooling layers for content and wording separately.  
\*\*\*Pooling at different levels: words, sentences, and paragraphs, then the results are pooled.  



#### LightGBM stacking with meta features

I stacked 9 models (Exp ID: 1 to 9) using LightGBM with the following meta-features:
- Number of characters in the summary text
- Number of characters in the prompt text
- Number of unique words overlapping between summary and prompt text (excluding punctuation)
- Number of overlapping words between summary and prompt text
- Ratio of overlapping bigrams
- Ratio of overlapping 4-grams
- Cosine similarity between TF-IDF values of summary and prompt text
- Mean of edit distances between each summary sentence and prompt sentence
- Number of words in the summary text (excluding punctuation, correcting misspelling)
- Number of unique words overlapping between summary and prompt text (excluding punctuation, correcting misspelling)
- Number of overlapping bigrams between summary and prompt text (excluding punctuation, correcting misspelling)
- Ratio of overlapping bigrams (correcting misspelling)
- Number of similar summary texts (similarity calculated by cosine similarity using Universal Sentence Encoder embedding)
- Readability score of the prompt text


This stacking model achieved CV: 0.478.


#### Averaging the Two Models
For the final submission, I simply averaged the stacking model and the long-token model (Exp ID: 10), and submitted it. 
The CV is 0.471, Public LB: 0.433. 
I also tried nelder-mead weight optimization, but CV was not well as the averaged score.


## Details of the submission

### What Worked

#### Model for Summary Text and Prompt Text Input

I initially concatenated the summary text and prompt text and fed them into deberta-v3-large model. However, this didn't lead to an improvement in CV performance.  
I realized it was necessary to inform the model about the positions of the summary text tokens. Therefore, I prepared attention masks to indicate the positions of the summary text, title, question, and prompt text within the input tokens.


These masks were used in conjunction with deberta's output. The architecture of this model is described below.
(I apologize if it seems overly complex. According to other solutions, it appears that using an attention mask solely for the summary text may be sufficient.)


This model was trained with a token length of 1800. However, during prediction, I had to reduce the token length to 1024 due to a 9-hour time limit.  
This model (Exp ID: 10) imporoved CV scores, especially challenging prompts, resulting in CV: 0.504, Public LB: 0.461, inference time exceeded 4h~.

### What didn't Work

#### CNN Model to the Sentence Similarity Matrix between Summary Text and Prompt Text
I noticed that significant errors were caused by summary texts that simply copied and pasted many sentences from their prompt texts. These summaries were given the same low content and wording score. I believe that when scorers come across such a copy-and-paste summary, they tend to stop reading it and automatically assign low scores. Therefore, I wanted to address this issue by calculating the edit distances between each of the N sentences in the summary text and each of the M sentences in the prompt text. I then created an N x M similarity matrix, with each element containing edit distance ratio.  
  
This matrix was fed into a CNN model, and target values were generated. However, this CNN model was not as effective as anticipated... (this idea was inspired by a method described in [this](https://www.cl.cam.ac.uk/~ek358/Summarization_NAACL.pdf) paper).



### Souces
- https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369956
    - For information on the multi-scale approach.
- https://www.cl.cam.ac.uk/~ek358/Summarization_NAACL.pdf
    - Reference material for tackling the same task.
- https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-feature-engineering
    - I borrowed many features from this source. Thank you very much for your contribution!
