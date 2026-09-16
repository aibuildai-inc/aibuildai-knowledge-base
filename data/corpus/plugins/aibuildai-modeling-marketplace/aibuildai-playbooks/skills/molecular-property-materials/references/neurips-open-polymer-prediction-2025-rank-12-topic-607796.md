# Private LB 0.083 | GNN Embeddings + Stacking Ensemble

Competition: neurips-open-polymer-prediction-2025
Rank: #12
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/private-lb-0-083-gnn-embeddings-stacking-ensemble

Hi. This is my solution if you care
# Context
- Business Context: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/overview
- Data Context: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data
# Overview

# Details
## GINEConv
[GINEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html) is a PyTorch Geometric layer. `GINEConv` is nice because it allows **juicy** edge features, while vanilla `GCNConv` only allows node features.
## Prepare data for GNN training
- Total node features = 40, edge features = 23.
- I added a global node, which connected to other nodes. The idea is to create shortcuts in the graph so information can flow more easily.
- I applied masking and no augmentation used.
## GNN model
```
class GINENet(torch.nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, first_conv_dim=107, second_conv_dim=107):
        super().__init__()

        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(node_dim, first_conv_dim),
                nn.ReLU(),
                nn.Linear(first_conv_dim, first_conv_dim)
            ),
            edge_dim=edge_dim,
            train_eps=True
        )
        self.ln1 = nn.LayerNorm(first_conv_dim)
        
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(first_conv_dim, second_conv_dim),
                nn.ReLU(),
                nn.Linear(second_conv_dim, second_conv_dim)
            ),
            edge_dim=edge_dim,
            train_eps=True
        )
        self.ln2 = nn.LayerNorm(second_conv_dim)

        self.lin1 = nn.Linear(second_conv_dim, first_conv_dim)
        self.lin2 = nn.Linear(first_conv_dim, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.ln2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.005)

        x = global_mean_pool(x2, batch)
    
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return x
```
I used `LeakyReLU` instead of `ReLU` after the last conv layer, so stacking models don’t get all-zero embeddings.
I trained this model with `Huber loss`, picked the best checkpoint by validation `MAE` on 3500 samples from competition data. I tuned this architecture’s hyperparameters and settled on the one that scored best on the public LB (0.069).

## Stacking Gradient Boosting Models
Now in addition to descriptors and fingerprints data, we have **juicy** GNN embedding and GNN preds to use 😋
I dropped low varience and high nan columns
| Feature Group   | Dimension                          |
| --------------- | ---------------------------------- |
| Descriptors     | 220                                |
| Fingerprints    | 162 (Truncated SVD from 1024 bits) |
| GNN Embedding   | 107                                |
| Graph Features  | 10                                 |
| GNN Predictions | 5                                  |
| **Total**       | \~500                              |

For each target I trained 5 fold model, used optuna to optimize
Finally I used random forest as a meta model

## Feature Importances (total, not average)
### XGB
| Target | Descriptor | Fingerprint | GNN   | GNN\_Pred | Graph |
| ------ | ---------- | ----------- | ----- | --------- | ----- |
| Tg     | 23.49      | 22.09       | 51.20 | 0.63      | 2.58  |
| FFV     | 27.55      | 5.15        | 66.45 | 0.12      | 0.73  |
| Tc     | 23.68      | 27.09       | 46.38 | 0.60      | 2.24  |
| Density    | 19.15      | 4.42        | 75.78 | 0.07      | 0.57  |
| Rg     | 29.23      | 19.42       | 50.33 | 0.41      | 0.61  |

### LGB
| Target | Descriptor | Fingerprint | GNN   | GNN\_Pred | Graph |
| ------ | ---------- | ----------- | ----- | --------- | ----- |
| Tg     | 14.77      | 24.75       | 57.74 | 1.22      | 1.51  |
| FFV     | 12.77      | 18.79       | 66.74 | 1.36      | 0.34  |
| Tc     | 12.73      | 26.12       | 56.92 | 2.44      | 1.78  |
| Density    | 20.28      | 13.78       | 63.95 | 1.69      | 0.30  |
| Rg     | 16.42      | 32.41       | 46.49 | 3.91      | 0.78  |

### Cat
| Target | GNN   | Descriptor | Fingerprint | Graph | GNN\_Pred |
| ------ | ----- | ---------- | ----------- | ----- | --------- |
| Tg     | 86.91 | 5.59       | 5.49        | 1.97  | 0.06      |
| FFV     | 93.97 | 3.73       | 2.09        | 0.15  | 0.07      |
| Tc     | 82.56 | 10.11      | 6.02        | 1.27  | 0.03      |
| Density    | 90.23 | 6.46       | 3.13        | 0.14  | 0.05      |
| Rg     | 85.11 | 9.07       | 5.45        | 0.26  | 0.11      |

## Top 30 Feature Importances For Target `Tc`
### XGB

### LGB

### Cat

# Intresting Twitches
## Things I did (might matter or not)
- I used external data: [Tc_SMILES](https://www.kaggle.com/datasets/minatoyukinaxlisa/tc-smiles) and [smiles_extra_data](https://www.kaggle.com/datasets/dmitryuarov/smiles-extra-data), but prioritize competition data by loading them first
- I dropped some outliers
- I tried `MAE` and `MSE` for GNN loss at first, both worked fine, I couldn't decide so I just chose `Huber`. All boosting models used `Huber` too.
## Things I tried but didn't work
- Multiple global nodes → overfit (1 was enough).
- Injecting descriptors into the global node or inside GNN layers → overfit.
- Pretraining on subgraphs with contrastive loss → not really helpful.
## Things I want to try
I just found out about `TabPFN` in the discussion section and believe it can help my stacking even more, but couldn't finish in time 😐
# Thanks for sticking till the end
