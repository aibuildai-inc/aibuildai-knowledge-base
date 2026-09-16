# 5th Place Solution: GNN with Invariant Dimension Features

Competition: predict-ai-model-runtime
Rank: #5
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456093

Thanks for hosting the interesting competition, and congratulations to the winners!

## Overview
My solution is based on an end-to-end graph neural network (GNN). I implemented a 3-layer GraphSage based on [PyG](https://pytorch-geometric.readthedocs.io/en/latest/). In each layer, I operate graph convolution in both directions of edges by different weights and concatenate the outputs.

I trained the model to minimize pairwise hinge loss using the AdamW optimizer using a cosine annealing scheduler.
For the loss, I used the average of a pairwise hinge loss among different configurations of the same graph and a pairwise hinge loss among all the samples in a batch (including different graphs). For this reason, I didn't use a subgraph but a whole graph as an input to GNN.

## Dimension Feature Embed by Transformer
Node features include 30 features (including tile and layout configurations) for each of the 6 dimensions. A naive approach to input this to GNN is to simply flatten them (I call this naive model), but I considered the following two disadvantages.
- It drops prior information about feature correspondence across dimensions
- The output should be invariant to the indexing order of dimensions (I'm not sure if this is exactly correct)

To tackle these issues, I implemented a dimension feature embedding layer using a transformer that handles each dimension as a token. In this layer, I transform (6, 30) input to (6, mid_ch) by a transformer and reduce to (mid_ch) by taking the sum in the token dimension.
Since most dimension features are exactly the same (padded ones), I could compute this efficiently by calculating embedding for only unique ones in each batch and copying them.

## Tile Config Dataset
I trained the model using only the tile dataset.
Using the transformer model, I could easily achieve 0.2 (nearly perfect) in public and private LB. The transformer model was significantly better than the naive approach on the validation Kendall tau score.

## Layout Config Dataset
I trained the model using the whole layout dataset (random and default of xla and nlp). Also, including the tile dataset enhanced the performance a little.

I could not outperform the naive model by the transformer model in the validation score (due to limited time), but it was comparable. My final submission was an ensemble of naive models and transformer models.

## Tips
- use the same opcode embedding for unary operations such as abs, ceil, cosine, etc.
- override layout_minor_to_major by layout config features for configurable nodes
- [DropEdge](https://arxiv.org/abs/1907.10903)
- apply log transformation to input features
- oversampling
- load layout config data by numpy's mmap mode to save RAM

## What Didn't Work
- graph pooling
- pretrain on the tile dataset and finetune on the layout dataset
- graph normalization
- dropout node
- GAT, GATv2, GIN
- fp16
- pseudo label

## Acknowledgement
I acknowledge Preferred Networks, Inc. for allowing me to use computational resources.


Source Code: https://github.com/knshnb/kaggle-tpu-graph-5th-place
