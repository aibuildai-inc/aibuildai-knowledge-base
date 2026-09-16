# 6th Place Solution

Competition: kaggle-llm-science-exam
Rank: #6
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/447647

Thanks to Kaggle for hosting the LLM competition - it was both challenging and well-structured. A big shout-out to our community for the insightful discussions and for demonstrating what's possible under limited GPU resources. Congratulations to the winning teams, we learned a lot from the solution write ups. Many thanks to my super talented teammates @ubamba98, @nbroad, @trushk and @abhishek for such an amazing collaboration and perfect teamwork!

## Code Links
* Github repo: https://github.com/rbiswasfc/llm-science-exam 
* Inference notebook: https://www.kaggle.com/code/ubamba98/a08-exam-solver?scriptVersionId=145861577

## Summary

From the very beginning, we followed the retriever-reader framework for answering the STEM MCQs.

- Retrievers: Get context data
    - Corpus: we created a custom STEM corpus by filtering wiki articles based on their category metadata
    - Generated synthetic MCQs using GPT-3.5, GPT4, LLaMA2 70b & Falcon 180b, which were used to train both retrievers and readers (downstream MCQ models)
    - Reranked top 10 retrieved chunks using a cross-encoder and provided top 2 to 4 re-ranked chunks to readers to solve the MCQs.
- Readers: Solve MCQs
    - We used an ensemble of DeBERTas and LLMs.
    - DeBERTa: we explored different training strategies, such as
        - Span approach: to make use of cross-option information
        - DebertaV2ForMultipleChoice: for diversity
        - [PET](https://aclanthology.org/2021.eacl-main.20/): for diversity
    - LLM: we explored both fine-tuning and zero-shot strategies
        - Finetuned Open-Orca/Mistral-7B-OpenOrca using LoRA
        - Top 5% of MCQs (as ranked by prediction entropy - proxy for MCQ difficulty) were handled by to 70b LLMs:
            - platypus2-70b-instruct
            - sheep-duck-llama-2

**NOTE**: We used `platypus2-70b-instruct` in our pipeline - We're uncertain about the compatibility of this model with the competition rules.

## 1. Retrievers: Get Context Data

We adopted the standard retrieve & re-rank pipeline to find MCQ specific relevant text chunks from a custom STEM wikipedia corpus.

### 1.1 STEM Wiki Corpus

To address frequent rendering issues (number, equations & symbols) and to filter out irrelevant articles from the existing wiki corpuses, we created a custom STEM wiki corpus as below:

- Define a set of seed wikipedia categories related to STEM topics such as Category:Concepts in physics, Category:Physical quantities, etc.
- For each category, recursively collect the member pages and subcategories up to a certain depth.
- Extract the page contents of the collected wiki URLs Wikipedia-API (~500k pages).

**Chunking:** We first split the full text from each article based on different sections. The longer sections were further broken down into smaller chunks containing approximately 300 tokens. We maintained two representations for each chunk:

- A short representation without overlap - used for embedding and search
- A longer representation with overlap from the previous and next chunks - used as contexts in the downstream MCQ models.

### 1.2 Retriever

Retrieved text chunks from pre-trained embedding models such as thenlper/gte-base, BAAI/bge-base-en-v1.5 provided a strong performance boost for the downstream MCQ models. In our pipeline, we observed further performance improvement by fine-tuning the embedding models. We re-purposed the synthetically generated MCQs for the retrieval task, as illustrated in the figure below:



Retrievers were fine-tuned with the NCE loss + dynamically computed in-batch hard negatives. Specifically, we used the batching strategy from this [solution post](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/395110).  In our pipeline, we used a union of top 10 text chunks retrieved from fine-tuned `thenlper/gte-base` & `BAAI/bge-base-en-v1.5` models.

### 1.3 Reranker

To reduce noise in the contexts provided to the downstream MCQ models, we re-ranked top retrieved candidates using a cross-encoder (deberta-v3-base). We selected top 2 to 4 re-ranked contexts as per a dynamic threshold on computed relevancy score from the re-ranker.

## 2. Readers: Solve MCQs

We explored several strategies to solve the MCQs using the retrieved contexts.

### 2.1 Span Classification

- Mean pooling of tokens from each option to extract option-wise features
- Classification with cross-option information



### 2.2 LLM Fine-tuning using LoRA

Towards the end of competition, it was evident that we were hitting performance ceiling with DeBERTa models. Hence, we decided to explore fine-tuning of LLMs. While we experimented with several models  (e.g. flan-t5-xl (3b), flan-t5-xxl (11b), llama-7b, mistral-7b, llama-30b and llama-70b), our final pipeline include 1x LoRA fine-tuned mistral-7b model. The lora settings were as follows:

- lr 3.e-4 (slight variations)
- rank 16
- alpha 16
- lora modules on all linear components (qkv, out_proj, ffn, gate_proj, etc.) and embeddings
- dropout 0.1
- no sequence packing
- bf16

We followed the cross-option setting during LLM fine-tuning i.e. to predict the right answer given all the options. We did not explore training the model in binary classification set up. 

### 2.3 Zero Shot using 70b models

A huge shout out to @simjeg for demonstrating the potential of 70b LLM using Kaggle GPU!!

About 5 days before the end of the competition, we started experimenting with 70b models. We realized we wouldn’t be able to run every example though a model this big, but we also knew that our deberta models were doing a good job for 90-95% of the questions. Thus, we took the least confident predictions and passed them to the 70b model, using @simjeg's code. We did slight prompt engineering the last few days and found that taking the difference between two tokens (yes/no or true/false) gave slightly better results. In the end we used both platypus 70b and sheep-duck-llama-70b. We adjusted the prompt from the public notebook to match the way it was trained. Our last few submissions were experiments passing 5-10% of the samples, blending with other models, or just having the 70b predictions be the final predictions.  We did get lucky because our best scores came on the final day.

CV on our custom dataset showed slightly better performance passing all the options at once, but we couldn’t figure out how to do that on Kaggle.

Local validation results with `Sheep-duck-llama`:
* (all options at once) ABCDE tokens: 0.91
* (one option at a time) only yes token: 0.886
* (one option at a time) only no token: 0.888
* (one option at a time) yes token - no token: 0.891

We did try fine-tuning the 70b model using QLoRA, but we ran into issues we could not debug before time ran out.

## Team Members
- Abhishek Thakur (@abhishek)
- Trushant Kalyanpur (@trushk)
- Udbhav Bamba (@ubamba98)
- Nicholas Broad (@nbroad)
- Raja Biswas (@conjuring92)
