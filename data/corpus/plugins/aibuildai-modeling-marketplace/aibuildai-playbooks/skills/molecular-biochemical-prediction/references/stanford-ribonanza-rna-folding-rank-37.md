# [37th place solution🥈] Single Graph Transformer + Gated GCN Model

Competition: stanford-ribonanza-rna-folding
Rank: #37
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/463253

# 1. Leadboard score
Public score: 0.15187; Private Score: 0.15349
# 2. Data preprocess

## 2.1 Graph node feature
input node feature:
```python
RNA_ONE_HOT = {
    'A': [1, 0, 0, 0],
    'U': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'C': [0, 0, 0, 1]
}
```
output node feature: DMS_MaP and 2A3_MaP score, which were clipped to [0, 1] when training.
## 2.2 Edge feature
For neighbor edge in original RNA sequence, the edge feature = 1.0;
For Ribonanza position pairs edge, the edge feature = 1.0 + 10 * Watson-Crick base pair probabilities, which are recorded in the supply Ribonanza_bpp_files.
# 3. Model
The model I chose was GraphGPS: github:[https://github.com/rampasek/GraphGPS](https://github.com/rampasek/GraphGPS)
## 3.1 Basic layer
The basic GraphGPS layer is composed of Graph Transformer and Gated GCN.

## 3.2 Positional encoding
I tried **Laplacian Positional Encoding** at the begining, but it didn't improve the performance of GraphGPS, which may means the Laplacian Positional Encoding is not useful when facing with topology diversity.
Hence, I removed the positional encoding in GraphGPS.
## 3.3 Model hyper-parameters
```yaml
model:
  type: GPSModel
  loss_fun: l1
gt:
  layer_type: CustomGatedGCN+Transformer
  layers: 16
  n_heads: 8
  dim_hidden: 256
  dropout: 0.1
  attn_dropout: 0.1
  layer_norm: False
  batch_norm: True
gnn:
  head: inductive_node
  layers_pre_mp: 0
  layers_post_mp: 3
  dim_inner: 256
  batchnorm: True
  act: relu
  dropout: 0.0
  agg: mean
  normalize_adj: False
optim:
  clip_grad_norm: True
  optimizer: adamW
  weight_decay: 1e-5
  base_lr: 0.001
  max_epoch: 50
  scheduler: cosine_with_warmup
  num_warmup_epochs: 3
share:
  dim_in: 4
  dim_out: 2
  num_splits: 3
  edge_dim_in: 1
```
## 3.4 Data augmentation
I tried two data augmentation methods: **sub graph sampling** and **RNA sequence reversing**.
To be specific, the sub graph sampling is choosing a random center node from original graph, and sampling its k-steps neighbors to generate a new graph; RNA sequence reversing means random reversing the input sequence when training.
Experimentally, I found the **rna sequence reversing** is more work for this mission.
