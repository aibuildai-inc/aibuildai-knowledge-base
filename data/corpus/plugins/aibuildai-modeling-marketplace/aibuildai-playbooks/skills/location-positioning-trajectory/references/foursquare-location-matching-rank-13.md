# 13th place solution (GNN)

Competition: foursquare-location-matching
Rank: #13
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/336124

First of all, this competition would have been better without the leak.
It is very sad that because of the leak, we can no longer distinguish between a good solution and a overfitting solution.

# Summary



My solution consists of four stages

- Candidate generation
  - 1st stage: Create candidates
  - 2nd stage: Filtering by LightGBM
- Matching
  - 3rd stage: match prediction by xlm-roberta-base and mdeberta-base-v3
- PostProcessing
  - 4th stage: Node Classification by GNN

# Cross validation strategy

GroupKFold(n_splits=2, group='point_of_interest')
The CV score of my solution is 0.920

# 1st stage

- Generates 60 candidates per id

  - 30 candidate neighborhoods per id using cosine similarity of text embedding by Tfidf
      - text: name, categories, address, city, state in a one sentence
  - 30 candidate neighborhoods by haversine distance for per id

# 2nd stage

- Reduce to 3.7M pairs with high match probability using LightGBM
- Features
  - Missing value or not(name, cat)
  - jaro(name, categories, address, city, state, zip, url, phone)
  - leven(name, categories, address, city, state, zip, url, phone)
  - haversine
  - cos sim(tfidf embedding)
  - mean top_k haversine, tfidf (k=5, 10, 15)
- Using ForestInference
- Max IoU: 0.971

# 3rd stage

- Predicts matche using a language model like DITTO
- model: xlm-roberta-base and mdeberta-base-v3
- Input
  - text: name_1 + [SEP] + name_2 + [COL] + categories_1 + [SEP] + categories_2 + [COL] + address_1 + [SEP] + address_2 + [COL] + city_1 + [SEP] + city_2 + [COL] + state_1 + [SEP] + state_2 + [COL] + zip_1 + [SEP] + zip_2
  - numerical_feature
    - haversine
    - mean top_k haversine(k=5, 10, 15)
- Training method
  - multitask learning: match and country (+0.003)
  - augmentation
```
import nlpaug as naw
    aug_list = [
  naw.RandomWordAug(action='swap', aug_p=0.1), 
  naw.RandomWordAug(action='delete', aug_p=0.1), 
  naw.SplitAug(aug_p=0.1)
]
```
- flip id_1, id_2
- TTA: flip id_1, id_2

# 4th stage

This stage is the most important part of my solution.
By using GNN, both CV and LB are improved by about 0.02x.  

Predicts the matching id for id_1 using GNN.

1. Create a 2-hop-subgraph for each id_1(max iou: 0.993)  
__example__ , Suppose the following table is given and we predict the id matching A.

| id_1 | id_2 | 
| ---- | ---- | 
| A    | B    | 
| B    | A    | 
| B    | C    | 
| B    | D    | 
| C    | D    | 
| D    | B    | 
| E    | A    | 
| E    | F    | 
| F    | E    | 

In this case, the 2-hop-subgraph for A is as follows(Think in terms of undirected graphs)



2. Predict whether each node matches id_1
3. Nodes that exceed the threshold are set as matches for id_1

## Details

- Model
This model is based on the top solution of the Japanese competition platform atmaCup.

```
class SimpleGCN(nn.Module):
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                #  num_classes: int,
                 deg):
        super(SimpleGCN, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        towers = 4
        pre_layer_num = 2
        post_layer_num = 2
        divide_input=False
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        conv = PNAConv(in_channels=num_node_features, out_channels=32,
                                    aggregators=aggregators, scalers=scalers, deg=deg,
                                    edge_dim=num_edge_features, towers=towers, pre_layers=pre_layer_num, post_layers=post_layer_num,
                                    divide_input=divide_input)
        self.batch_norms.append(BatchNorm(32))
        self.convs.append(conv)

        for j in [32, 64, 128]:
            conv = PNAConv(in_channels=j, out_channels=j*2, aggregators=aggregators, scalers=scalers, deg=deg, edge_dim=num_edge_features, towers=towers, pre_layers=pre_layer_num, post_layers=post_layer_num, divide_input=divide_input)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(j*2))

        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(j*2, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = self.dropout(x)
        x = self.linear(x).flatten()
        return x
```

- node features
  - marker(I named it): 1 if the node corresponds to an id_1, 0 if it is a candidate.  
  The presence of this feature allows us to solve for node classification rather than link prediction.
  - Missing value or not(name, categories, address, city, state, zip, url, phone)
- edge features
  - mdeberta pred
  - xlm-roberta pred
  - lgbm pred
  - jaro (address, city, state, zip, url, phone, categories)
- loss function
  - IouLoss for each graph and the binary cross-entropy of the entire batch are the loss functions

```
def iou_loss(logits, data):
    sizes = degree(data.batch, dtype=torch.long).tolist()

    logit_list = logits.sigmoid().split(sizes)
    target_list = data.y.split(sizes)

    loss = 0
    for y_pred, y_target, group_size in zip(logit_list, target_list, data.group_size):

        intersection = (y_pred * y_target).sum()
        total = (y_pred + y_target).sum() + (group_size - y_target.sum())
        union = total - intersection
        loss += 1 - (intersection / union)

    return loss / data.num_graphs

loss = iou_loss + 0.1*BCELoss
```

## Advantages of GNN
  - Creating a 2-hop-subgraph greatly improved maxiou.
  - Iou loss made it possible to optimize the competition metric directly.
  - Fast inference

# Public Score Timeline

1. 20 candidates per id and mdeberta and lgbm stacking: 0.907
2. 60 candidates per id and xlm-roberta-base+mdeberta and lgbm stacking : 0.924
3. 60 candidates per id and xlm-roberta-base + mdeberta and GNN Post Processing: 0.946
