# 28th-place solution

Competition: kaggle-llm-science-exam
Rank: #28
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446610

First of all, thanks to competition organizers for hosting this competition. I would also like to congratulate everyone who made it through this tough competition to the end.

Unfortunately my solution overfits the public lb and I missed the gold medal, but I would like to share a summary of my solution here.

## Wikipedia data for retrieval
As reported in [this discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/442595), there're cases in which some numbers or even paragraphs are missing from the wikipedia data, so I created whole wikipedia plain text from [CirrusSearch dumped data](https://dumps.wikimedia.org/other/cirrussearch/).

## Context retrieval
I combined two types of retrieval methods.

### 1. bi-gram tfidf inverted index
For each wikipedia article, I created a 2-gram tfidf based inverted index. I referred [this repository](https://github.com/GINK03/minimal-search-engine) to create the inverted index. Indexing all articles resulted in slow search speeds, so I indexed about half of the articles based on their creation date and popularity.

#### 1st step
Using prompt and answer as query, extract some wikipedia articles with high similarity from the inverted index database. The example of retrieval notebook is [here](https://www.kaggle.com/code/daisuketakahashi/wikipedia-bi-gram-inverted-index/notebook?scriptVersionId=146229927).

#### 2nd step
The extracted wikipedia articles are split into sentences, and the similarity with the query is calculated by tfidf to extract some sentences. The three sentences before and after the extracted sentence were added as the context.

### 2. faiss
The full text of wikipedia was split into sentences so that each chunk is about 25 words. Each chunk was vectorized by [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and a faiss index was created. The faiss index was quantized and compressed to about 7 GB.

#### 1st step
Using prompt and answer as query, extract some chunks with high similarity from the faiss index.
#### 2nd step
The extracted chunks and the chunks before and after were used as context.

## Model
Almost the same as a baseline notebook.
I trained two deberta-v3-large model with different training dataset.
The results were inferred by using the following 7 contexts. These results were ensembled by simply averaging each output.
- two contexts from bi-gram tfidf inverted index
- five contexts from faiss index
