# 19th-place solution

Competition: kaggle-llm-science-exam
Rank: #19
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446395

Thank you Kaggle for hosting this challenging competition on LLM . 
A special thanks to Kagglers who shared their work, making the competition even more stimulating. 
As usual, congratulations to the winners and everyone who enjoyed the competition!

Here is a write-up of my solution

## Context Retrieval
It's a two-step process:
- In the first step, I generate candidates through an indexed search across all Wikipedia paragraphs.
- In the second step, I rerank the candidates using a TF-IDF model to fully leverage the fact that prompts/answers use same words of wikipedia sentences: This second step will prove to be very important for the final score.

For index generation, I used the Hugging Face dataset [wikipedia_20220301.en](https://huggingface.co/datasets/wikipedia):  I removed all sections such as 'See also,' 'References,' 'External links', ...  and split the text by `\n\n`. 
Additionally, I eliminated all paragraphs with fewer than 10 words.

I generated the embedding for each paragraph using 'all-MiniLM-L6-v2' with 'MAX_LENGTH = 512' and created 10 Faiss indexes,  each containing 4.2 million rows

The sequential search of these indexes on the GPU T4 x2 instances is relatively fast (less then 10 minutes for 200 prompts)

You can find the dataset and an example of a search here: [Semantic search of Wikipedia EN with Faiss](https://www.kaggle.com/code/steubk/semantic-search-of-wikipedia-en-with-faiss)

## LLAMA2 70B Inference

I used 'orca-mini-v3-70b' and the amazing job done by @simjeg to run LAMA2 70B on GPU T4 x2..

This is the prompt:

```
### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.

### User:
Your task is to analyze the question and the proposed answer below. If the answer is correct, respond with yes; otherwise, respond with no. As a potential aid, context from sentences of Wikipedia pages delimited by ## is available, even if it may not always be relevant.
Context: {row['context']
Question: {row['prompt']}
Proposed answer: {row['A']

### Assistant:
```

## ModelForMultipleChoice

I trained Deberta V3 (max_length = 450) with `AutoModelForMultipleChoice` on a dataset of 24k prompts extracted from the dataset  [llm-science-exam-dataset-w-context](https://www.kaggle.com/datasets/mgoksu/llm-science-exam-dataset-w-context) by @mgoksu, taking only the prompts that start with 'What,' 'Which,' 'How,' 'Who,' and removing duplicate prompts from the training set. 
I then validated it on the 200 rows training set.

## Submission
Final submission is merge of softmax of the 2 models.

To minimize memory leaks, each step was executed in a separate script.

Submission time of the pipeline time varies from 7 hours and 20 minutes to the timeout (in the last days!!) 

Below is the contribution of each step to the final score:

**Single ModelForMultipleChoice Model**: 
LB: 0.822, Private: 0.805

**Single ModelForMultipleChoice Model with first step context retrivial**:
LB: 0.889, Private: 0.884

**Single orca-mini-v3-70b Model with first step context retrivial**:
LB: 0.897, Private:	0.889

**Merge ModelForMultipleChoice + orca-mini-v3-70b with first step context retrivial**:
LB: 0.906, Private: 0.901

**Single orca-mini-v3-70b Model with two steps context retrivial**:
LB: 0.904, Private:	0.906

**Merge ModelForMultipleChoice + orca-mini-v3-70b with two steps context retrivial**:
LB: 0.921, Private:0.917

Link to Inference notebook: https://www.kaggle.com/code/steubk/llm-mc-qa-openbook-v3-inf-val-tfidf-db450
