# 7th Place Solution

Competition: kaggle-llm-science-exam
Rank: #7
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/447155

Thanks to Kaggle staff for organizing the brand new and interesting competition.
I really learned a lot of new things through this competition, including RAG and fine-tuning of LLM. I would like to thank all those who contributed meaningful discussions and published high-level notebooks during the competition.
I would also like to thank my teammates @thedrcat, @anonamename and @kashiwaba for refining our solution with a variety of ideas.

# Overview

- Use contexts created from various wikipedia data.
- Ensemble of DeBERTa-V3 large models and LLMs.
- Multi-stage strategy: incorporating various combinations of contexts and models by reducing the number of questions to be predicted at the later stage.



# Retrieval part

### Wikipedia-20230801 (article retrieve -> sentence retrieve)
This search strategy is based on the method used in JJ's [public notebook](https://www.kaggle.com/code/jjinho/open-book-llm-science-exam), but with the following changes:

- We use [Wikipedia-20230801-dump](https://www.kaggle.com/datasets/bwandowando/wikipedia-index-and-plaintext-20230801/versions/2). This contains more articles (6.67M) than [Wikipedia Plaintext (2023-07-01)](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) (6.27M).
- For the article retrieve, we used the embeddings consisting of the full text of the article (for each article, we split all text in an article into chunks of 256 tokens).
- We use gte-small for creating the above embeddings.
- We have modified the sectionize_documents function for better sententialization.
- We use TF-IDF for the sentence retrieve.

### MB 270K + TF-IDF

We use the MB's 270K dataset and TF-IDF method from [MB's great notebook](https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles).

### MB 270K + sentence-transformer

We search for MB's 270K text using gte-small with max_seq_length=512

### Cohere TF-IDF

This method is based on the hypothesis that paragraph-based search and sparse method may be better than dense methods specifically for the competition data. The questions and options often contain very specific terms which might not be well represented via dense embeddings. Also, paragraph-based search is generally considered more effective for question answering. The challenge is how to effectively perform sparse retrieval on many millions of Wikipedia paragraphs.

- We start with Cohere en-wiki dataset via HF. It's not the latest dump, which might cause some problems as some articles changed, but it was conveniently split into paragraphs. It also suffers from the problem mentioned on the forums, some LUA numbers/expressions are not properly parsed. 
- We use BERT tokenizer and vocabulary for TF-IDF to keep a reasonably sized vocab. We train TF-IDF vectorizer on the first 10M paragraphs and use that vectorizer for all paragraphs.
- We precompute indexes and store both indexes and paragraphs in kaggle datasets, in 4x batches of 10 million rows
- We use [fast sparse KNN lookup](https://github.com/ing-bank/sparse_dot_topn), co-developed and previously shared by Ahmet Erdem, to find top 5 nearest neighbors context paragraphs for each question. 

# Model part
 
## Validation

Initially, we used the samples provided in train.csv for validation, but after achieving CV ~ 0.99, it became challenging to assess the correlation between CV and LB. Therefore, we generated approximately 3000 additional samples by Chat-GPT and used these as new validation set. We used the [130K STEM articles](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/425106) as Chat-GPT input.
Although there is some blurring, CV and LB correlate relatively well, and the combination of models in the final ensemble was determined based on the CV in this validation set.



## DeBERTa-V3 large
Based on [cderotte's notebook](https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1), we train the following four models.
  * max_length=512, microsoft/deberta-v3-large (x2, we will refer to these as v1 and v2 in the Ensemble section)
  * max_length=512, deepset/deberta-v3-large-squad2 (x1, we will refer to it as v3)
  * max_length=1024, microsoft/deberta-v3-large (x1)

## LLM

### CausalLM
We used AutoModelForCausalLM class from transformers and SFTTrainer class from trl for LLM training.
#### Training
* We trained the model to output one of the letters A, B, C, D, or E with the following prompt
```
### Input: <context>\n\n### System: Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]. Use the input text above as a reference for your answers if needed.### Question: <prompt>\nA) <option A>\nB) <option B>\nC) <option C>\nD) <option D>\nE) <option E>\n\n### Answer:
```
* Use QLoRa for memory saving.
* Shuffling the choices during the training worked a bit.

#### Inference
* max_new_tokens=1
* We put the logits of the [A,B,C,D,E] tokens of the first output into softmax function to obtain the probability of each choice.

### CausalLM Reward Modeling

* Modified trl's RewardTrainer class to train AutoModelForCausalLM with Reward Modeling.  
* Input is in the following format
    `<context> #### <prompt> #### <option> #### `
* Generate pairs with the correct choice as "chosen" and any other choice as "rejected" (this is the same as normal Reward Modeling).
* Use the predicted value of the next "yes" token in the input string as the logit to be compared.
* LLMs that do not support AutoModelForSequenceClassification can now be trained with Reward Modeling, and can be combined with normal CausalLM models to ensemble more diverse LLM models.

### Single Model Results
context : Wikipedia-20230801
| Model | Type | Private LB | Public LB |
| --- | --- | --- | --- |
| Mistral-7B-v0.1 | CausalLM | 0.874 | 0.868 | 
| Mistral-7B-OpenOrca | CausalLM | 0.882 | 0.876 | 
| Mistral-7B-v0.1 | CausalLM Reward Modeling | 0.897 | 0.888 | 
| Mistral-7B-OpenOrca | CausalLM Reward Modeling | 0.896 | 0.888 | 
| OpenOrca-Platypus2-13B | CausalLM | 0.872 | 0.880 | 
| Llama2-chat-AYT-13B| CausalLM | 0.872 | 0.875 | 

# Ensemble

The problem with LLM is its long inference time. Therefore, we first use Deberta models to filter out easy questions, which we define as questions with a high prediction probability (max probability > 0.7), and then use LLMs to infer only remaining questions. In this way, the inference time is reduced by narrowing down the data to be inferred with LLM. We also applied this step-by-step inference to Deberta models to keep the inference within 9 hours. This ensemble method allowed us to add 3 LLMs to the 9 hours of inference. The final prediction is a weighted average of 8 Deberta models and 3 LLMs across different contexts. We assigned higher weights to LLMs than to Deberta models. The weights were determined from map3 of the validation set.

### Final submission
Private LB : 0.925, Public LB : 0.931
| Model | Context | Stage | Weight | 
| --- | --- | --- | --- |
| DeBERTa-V3 large (max_length=512), v1 | Wikipedia-20230801 | 1st | 1.0 | 
| DeBERTa-V3 large (max_length=512), v1 | MB 270K + TF-IDF | 2nd | 1.0 | 
| DeBERTa-V3 large (max_length=512), v2 | Wikipedia-20230801 | 3rd | 0.5 | 
| DeBERTa-V3 large (max_length=512), v2 | MB 270k + sentence-transformer | 3rd | 0.5 | 
| DeBERTa-V3 large (max_length=1024) | Wikipedia-20230801 | 3rd | 0.5 | 
| DeBERTa-V3 large (max_length=1024) | MB 270k + sentence-transformer | 3rd | 0.5 | 
| DeBERTa-V3 large (max_length=512), v3 | Cohere TF-IDF | 3rd | 0.5 | 
| DeBERTa-V3 large (max_length=512), v3 | MB 270K + TF-IDF | 3rd | 0.5 | 
| Mistral-7B-v0.1, CausalLM Reward Modeling | Wikipedia-20230801 | 4th | 1.25 | 
| Mistral-7B-v0.1, CausalLM Reward Modeling | MB 270K + TF-IDF | 4th | 1.5 | 
| OpenOrca-Platypus2-13B, CausalLM | Wikipedia-20230801 | 4th | 1.25 |
