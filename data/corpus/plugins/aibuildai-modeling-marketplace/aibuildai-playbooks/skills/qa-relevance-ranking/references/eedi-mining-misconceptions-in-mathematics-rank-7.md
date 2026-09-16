# Private 7th (Public 2nd) Place Solution Summary

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #7
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551388

First of all, I'd like to thank kaggle team, host for hosting this competition , and my team mates (@tereka @irrohas @aerdem4)
I finally got 5th gold medal. I'm really happy to be Competitions Grand Master.

## Overview

Voting Ensemble of 3pipelines (Retriever + Reranker)
All reranker inference with vllm 

## Pipeline1 by Masaya

This is brief summary.
Masaya will write up detail on this [link](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551430).

Retriever
- Generate test answer with pretrained qwen35B
- model :SFR-Embedding-2_R, Hard Negative Sampling(N=100) removed if unseen in training data.
- dataset: Filled unknown Misconception with GPT4o-mini
- candidate 60

Reranker
- Qwen2.5 32B AWQ+LoRA (with 40 options)
  - options → 0~9 + A~Z + a~z (40 charactors)
- dataset: only original
- candidate 40
- run twice and merge ([:40] and [20:]) 
- left 25 with logprobs

## Pipeline2 by Tereka+Kaeru

Retriever
- model1 :SFR-Embedding-2_R、Hard Negative Sampling(100）
    - dataset: original + synthesis with gemma 27B/Qwen2.5 32B
    - lb 0.378
- model2 : public model by sayoulala
- weighted avg model1 and model2
- candidate 70 and filter with masaya preds
- only single fold

Reranker
- Qwen2.5 32B AWQ+LoRA (with binary options)
  - options → yes, not
- dataset: original + synthesis unknown misconceptions with gemma2 27B/Qwen2.5 32B
- only single fold

## Pipeline3 by Ahmet

Retriever
- Omitted in final pipeline

Reranker
- Candidates are each top9 from previous pipeline
- Qwen2.5 32B (with 9 options)

## Ensembling
- Voting ensemble of each pipeline
- Apply rank like following, join, sum rank, sort by sum_rank

```python
sub0
  .with_columns(pl.col("MisconceptionId").str.split(" "))
  .with_columns(pl.lit([1/i for i in list(range(1, 26, 1))]).alias("rank"))
  .explode(pl.col(["MisconceptionId", "rank"]))
```

```python
submission
  .sort(by=["QuestionId_Answer", "sum_rank", "rank1", "rank2"], descending=[False, True, True, True])
  .group_by(["QuestionId_Answer"], maintain_order=True)
  .agg(pl.col("MisconceptionId"))
```

## Postprocessing
- Boost missing misconception rank

## Accelerating Tips

Retreiver
- Run parallel with cuda:0 and cuda:1
- Pre-calculating Misconception Vector
- Dynamic padding with collate_fn

Reranker
- Inference with vllm (AWQ 4bit quantization)
- Set vllm parameter enabling_prefix_cache True
- Filter candidate with previous preds of pipeline

## What not worked

Retrieval
- Concat vectors
- Avaraging many vectors

Reranker
- Full-data trained model
- QwQ model

Thank you!!
