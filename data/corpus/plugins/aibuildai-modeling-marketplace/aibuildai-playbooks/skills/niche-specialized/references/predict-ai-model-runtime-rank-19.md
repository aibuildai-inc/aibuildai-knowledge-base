# 19th Place Solution

Competition: predict-ai-model-runtime
Rank: #19
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456074

Thanks to Kaggle and Google for organizing this competition! This was my first competition and I really enjoyed it. We didn't know anything about GNN's before starting this competition and we learned a lot throughout this competition. I would also like to thank my great teammate @roysegalz 

### Tile dataset:
For the Tile dataset we used an RGCN where the relations are the node_opcode. The model consisted of 4 blocks where a block consisted of RGCN -> LayerNorm -> ReLU
* loss function - ListMLE
* hidden dim = 128
* CosineLRScheduler
* AdamW
* lr = 4e-4
* weight decay = 1e-4
* max aggr for final graph representation

We reached 0.198 on Tile + sample_submission (random predictions on Layout datasets)

### Layout datasets:
We realized that the best way to differentiate between different graphs is by emphasizing the node_config_feat. We recognized the problem that the number of nodes that contain node configs is relatively small compared to the number of nodes in the graph. Hence we understood that this data might get lost in the model throughout the forward pass. Our way to solve the problem was the following:

First we represent the node_config_feat in a different way (nn.Embedding and one hot vector). After doing that we created this model architecture (which is the most important part of our solution):


Concatenating the data again after each block increased our scores dramatically.

We used the same model architecture for all the different layout datasets.

The model consisted of 3 blocks where a block consisted of GATv2-> LayerNorm -> ReLU (We decided not to use RGCN since it was really computationally expensive to run on the layout dataset)
* hidden dim = 64
* loss function - nn.MarginRankingLoss(0.5) 
* CosineLRScheduler
* AdamW
* lr = 2e-4
* weight decay = 1e-4
* Virtual node for final graph representation


Our CV Scores:

| Dataset | Kendal-Tau CV |
| --- | --- |
| XLA default| ~0.3 |
| NLP default | ~0.5 |
| XLA random| ~0.62 |
| NLP random| ~0.94 |
