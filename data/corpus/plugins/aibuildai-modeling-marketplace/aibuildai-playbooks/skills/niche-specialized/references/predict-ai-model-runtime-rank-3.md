# 3rd place solution write-up

Competition: predict-ai-model-runtime
Rank: #3
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456377

First of all, thanks for hosting this competition. It was my first Kaggle competition which I entered on a whim because I had some free time. It turned out to be a lot of funstration (fun + frustration). Looking at all the amazing solutions from the other teams, I consider myself incredibly lucky to have ranked this high!

**Edit:** After cleaning up the code I noticed that I forgot to mention some things which I now added. The code is now available [here](https://github.com/jafluri/kaggle_tpu_graph).

## Overview

My solution is more or less composed of three parts. Minor feature extraction and engineering, tinkering with the graph, e.g. pruning, and training a graph neural network (GNN). The GNN layers are based on the [GPS layers](https://openreview.net/pdf?id=lMMaNf6oxKM), using SAGE convolutions, [Linformers](https://arxiv.org/pdf/2006.04768.pdf) and [learnable positional encodings](https://arxiv.org/pdf/2110.07875.pdf). Note that this discussion concerns mainly the layout dataset. The solution for the tile dataset is mentioned briefly at the end.

## Input Features

I used all 140 provided input features and used a simple log transform after shifting them such that each feature was at least 1. Additionally, I went through the protocol buffers and extracted the following features:

- `has_dynamic_com`: A flag indicating whether the graph has dynamic computations.
- `is_root_of_com`: A flag indicating if a node is the output node of a computation.
- `indices_are_sorted`: A flag that I am not sure why I added it.
- For the `dot` operation, I extracted `lhs_contracting_dimensions`, `rhs_contracting_dimensions`, 
 `lhs_batch_dimensions` and  `rhs_batch_dimensions`, which are all integer sequences that I padded to a length of 3, so 12 features in total.
- For the `gather` operation I added the integer sequences `offset_dims`, `collapsed_slice_dims` and 
 `start_index_map` padded to length 3, the single integer `index_vector_dim` and the sequence `gather_slice_sizes` padded to length 5.

The padding lengths were chosen based on the longest sequences contained in the dataset and I always used -1 as the padding value. Some of these features were useless depending on the applied graph pruning, but I left them in the input anyway. Additionally, while going through the protobufs, I added the input shapes (6D) of the two input arguments for the `dot` and `conv` operations as additional features, making sure that they are always ordered in the same way (e.g. lhs, rhs arguments for the `dot`). This adds 16 dimensions to the input (with sum and products of the shapes). I did this because I thought that it might be difficult for the network to learn the order of the inputs and dimensions which are reduced in these operations given solely the message-passing networks.
I also took the 30 dimensional features and added them to the input features once modulus 128 (`(x % 128)/128` and once as true divide `(x // 128)/10` with some normalization. I did this to make it easier for the network to process the dimensions of the tensors and compare them to the [register size](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/437673) of the TPUs.

## OP code embeddings and positional encoding

I used 128-dimensional embeddings for the OP codes. For the positional encodings, I used RWPE described [here](https://arxiv.org/pdf/2110.07875.pdf). I created 16 dimensional PEs with the directed adjacency matrix and 112 using the undirected one, for a total of 128 features. The encoding was always calculated with the full graph, independent of the pruning that was used during the training of the network.

## Graph Modifications

I experimented with three versions of pruning/pooling:

- Dropping all nodes and connections besides the configurable one. This results essentially in training an MLP.
- Dropping all the nodes besides the configurable nodes and their inputs/outputs.
- Merging all nodes besides the configurable nodes and their inputs/outputs. Two nodes were merged if they had at least one connection and were neither configurable nor an input or output of a configurable node. This was done until no further merging was possible. The merged nodes had a unique OP code but their features are set to zero. 

Addionally, I added a virtual output node, connecting all nodes that produce ouputs.

## GNN

The GNN consists of SAGEConvolutions and Linformer. The architecture is shown in the figure below. The SAGEConvolutions use both the input and the output nodes with different weights and a message dimension of half the size of the input dimension. The Linformer dimension was set to 128 (or 256 in some experiments). I used Sigmoid Linear Units (SiLU) activation functions and a lot of layer normalisation.



The training was done with Adam and a cosine annealing scheduler. I tried batch sizes of 8, 16 and 32 with 5, 8 or 10 configs and pairwise hinge-loss. I trained on all collections at the same time and then did finetuning on the individual collections. However, I did not have time to train a network on all collections with the merged nodes. I only implemented this towards the very end and trained only one network on the `xla:default` collection, which gave me the best CV. The final submission was composed of networks trained with my second pruning strategy for `xla:random` and the `nlp` collections and the third pruning strategy for `xla:default`.

# Tile Network

The tile network was a simple GNN with 5 SAGEConvolutions, no extra features and no positional encodings. 

## Other stuff

These are things I tried out but failed or could not evaluate if they had a consistent positive impact on the results. 

- Node dropout and test-time augmentations with the dropout. 
- Using the full graph
- Transformers instead of Linformers (Memory)
- Some self-implemented attention with configs and features
