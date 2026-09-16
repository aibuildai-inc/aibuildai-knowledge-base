# 8th Place Solution Overview

Competition: shopee-product-matching
Rank: #8
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238125

We thank all organizers for this very exciting competition.
Congratulations to all who finished the competition and to the winners.

## Summary

[shopee_overview]

- Validation
  - GroupKFold(k=5), group by label_group
- Embeddings
  - Concatenated Image, TF-IDF and Title embeddings (21208dim)
    - Image: CNN(ResNet101/152) + GeM + CosFace  (3328dim)
    - TF-IDF: TfidfVectorizer  (10200dim)
    - Title: BERT(distilbert-base-indonesian) + ArcFace  (7680dim)
- Brute-force kNN by Faiss on embeddings converted to fp16
  - Use αQE + DBA [1][2]
  - 0.59 Threshold for cosine similarity (local CV best threshold +0.15)
- Post processing
  - Only if no pair is found for an product, the threshold is ignored and the product is paired with its nearest neighbor.


## Model details

(Add more details later)

- Image Embedding
  - ResNet152x2, ResNet101x3 with GeM Pooling (5fold)
  - Loss: CosFace
  - Optimizer SGD lr=1e-3 WarmupCosineAnnealing LR Scheduling
  - Input size 512x512
  - Embedding dimension 512(ResNet152), 768(ResNet101)

- TF-IDF
  - Use scikit-learn's TfidfVectorizer
  - Preprocessing
      - Unicode handling
      - NFKC normalize
  - Embedding dimension 10200

- Text Embedding
  - distilbert_base_indonesian (5fold)
  - Concatenate mean of 4,5,6 Layer, CLS and mean of token embeddings (total 3840dim)
  - Add FC and Tanh activation to reduce embedding dimension from 3840 to 1536
  - Loss: ArcFace
  - Optimizer: AdamW lr=1e-4 WarmupLinear LR Scheduling
  - Embedding dimension 1536


## Change in Private/Public LB

|Embeddings        |αQE+DBA|Private LB|Public LB|
|:-----------------|:------|:---------|:--------|
|Image             |       |0.706     |0.714    |
|Image+TF-IDF      |       |0.738     |0.749    |
|Image+TF-IDF+Title|       |0.748     |0.759    |
|Image             |✔      |0.717     |0.725    |
|Image+TF-IDF      |✔      |0.751     |0.766    |
|Image+TF-IDF+Title|✔      |0.761     |0.775    |


## Reference

[1]: [End-to-end Learning of Deep Visual Representations for Image Retrieval](https://arxiv.org/abs/1610.07940)
[2]: [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512)
