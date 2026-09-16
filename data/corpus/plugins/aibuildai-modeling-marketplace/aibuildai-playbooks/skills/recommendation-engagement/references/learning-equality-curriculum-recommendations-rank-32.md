# 32th place solution

Competition: learning-equality-curriculum-recommendations
Rank: #32
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/395018

Thank you to the organizers for the fun competition and everyone who participated.
I share my solution.

# Summary
- 2-stage: Retrieval (Bi-Encoder) and Re-Ranker (Cross-Encoder)
- pipeline: below



# 1. Retrieval
- input data:
  - topic: `title + description + [SEP-Depth] + level + [SEP-context] + context + [SEP-children] + children`
  - context: `title + description + text + [SEP-Kind] + kind`
- split train/valid: StratifiedGroupKFold (y=channel, group=topic_id) only 1fold
- model:
  - Bi-Encoder (Sentence-Transformers)
  - loss: NT-Xent loss (https://arxiv.org/pdf/2002.05709.pdf)
  - pretrained-model
      - (1) xlm-roberta-base
      - (2) sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  - tokenizer: add special token ([SEP-Depth] etc.)
  - batch_size: 256, max_len=128

|pretrained-model | training data| Rec@10 | Rec@50 | f2@10 | pub@10 | pri@10|
|---|---|---:|---:|---:|---:|---:|
|xlm-roberta-base | train |76.8|91.1|50.3|46.9|46.9|
|paraphrase-multilingual-mpnet-base-v2 | train |78.5|91.5|51.5|47.2|47.4|
|paraphrase-multilingual-mpnet-base-v2 | train+valid |93.3|99.0|62.1|48.9|49.5|

# 2. Select Candidate
- compute embedding vector by model, and calculate cosine-similarity between all topics and all contents 
- select top50 by cosine-similarity per model-> select duplicate candidates
- top10 : public=53.4, private=55.4

# 3. Re-Ranker
- input data: 
```python
    title + description + [SEP-Depth] + level + [SEP-context] + context + \
    [SEP-children] + children + [SEP] + \
    title + description + text + [SEP-Kind] + kind
```
- split train/valid: same as stage 1
- model:
  - Cross-Encoder
  - loss: BCE loss
  - adversarial-learning: FGM
  - batch_size: 128, max_len=256
- thres: 0.1

|#|model| training data| local | public | private|
|---|---|---|---:|---:|---:|
|1| xlm-roberta-base |train       | 67.2 |61.3|64.1|
|2 |paraphrase-multilingual-mpnet-base-v2 |train       | 67.6 |61.8|64.7|
|3|paraphrase-multilingual-mpnet-base-v2 |train+valid | 68.0 |63.4|66.3|
|final |  ensemble (weight=1:1:3) | - | 69.3 |64.4|67.8|

# Did't work
- define graph data by topic's structure of curriculum, and train GNN (Link Prediction). But didn't work.
- use LightGBM in stage 2 (But higher team was using it, so my method was bad...)

Thank you for reading.
