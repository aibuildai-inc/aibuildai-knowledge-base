# 9th Place Solution: GNN with Compressed Graphs Using Dijkstra’s Algorithm

Competition: predict-ai-model-runtime
Rank: #9
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456206

We would like to express our sincere gratitude to our Kaggle teammates and to our hosts for providing the opportunity to participate in such an engaging competition. And thanks to my excellent teammate @yoichi7yamakawa. 

This was my 7th (and 4th with @yoichi7yamakawa) gold medal this year. Wow!

The problem posed was of a type that we had not tackled much before, which made it extremely fascinating to delve into. Here below we outline our solution.

# On Tiles
We referenced [GitHub - google-research-datasets/tpu_graphs](https://github.com/google-research-datasets/tpu_graphs) and public notebooks. 
Since the score from the public notebook was already satisfactory, the successful learning of the layout part was the key to this competition.

# On Layout
We mainly used the implementation of [GutHub - kaidic/GST](https://github.com/kaidic/GST/tree/main) as a reference and made modifications that we felt were necessary to improve the score.

## Graph Compression
During the learning of the layout, the configurable data was limited. Therefore, our implementation only extracted nodes influenced by this and their connected components. 
By applying this, our learning efficiency dramatically improved, significantly contributing to the improvement of the score.

Specifically, the following procedures were performed for each data set
1. An undirected graph was constructed using edge_index, and Dijkstra’s algorithm was applied starting from the node with the largest index `s` (i.e., `s=data["node_feat"].shape[0]-1`). We chose `s` as the starting point because many graphs were trees with `s` as the parent.
2. The shortest path from `s` to all the nodes in `node_config_ids` was calculated, and the union set of nodes and edges in the path was considered as a compressed graph.

This compression method reduced the average number of nodes for each layout data from 13894 to 1736 for xla and from 5711 to 570 for nlp. For features of nodes not included in the compressed graph, we simply ignored them completely.

## Preprocessing (log-transform)
There were times when the learning could not proceed because orders differed depending upon the dimensions of input features. 
To cope with this, we underwent log transformation for node features, which consequently made learning proceed smoothly.

## Training Strategy
- 512 configs were sampled for each data per iteration. The bach_size were 2 or 4.
- Trained separate models for the four data types xla-random, xla-default, nlp-random, and nlp-default for 1000 epochs.
- The pairwise hinge loss was used (same as original implementation).

### CV score
- xla random: 0.7
- xla default: 0.33
- nlp random: 0.96
- nlp default: 0.51

## Data Specialized for Specific Model Types
Upon examining the data, we found that architectures (such as BERT, U-Net, ResNet...) could be inferred from IDs or node numbers (or op_codes for test data).
 Leveraging this information, we performed learning using data related only to each architecture, which substantially contributed to improving the private score. In addition to the models trained on the entire dataset, we trained models focused on resnet, efficientnet, or bert. Proper EDA is indeed crucial.

## Multiple GNN Architectures
The TransformerConv and GATConv as Graph conv layers had minimal impact but were useful for ensemble purposes.

### Ensemble
Although we didn't have sufficient time for meticulous weight tuning, it certainly contributed to steady score improvement (+0.01 - 0.02).

### What Didn't Work
- MSELoss as aux-loss
- Learning using all nodes
