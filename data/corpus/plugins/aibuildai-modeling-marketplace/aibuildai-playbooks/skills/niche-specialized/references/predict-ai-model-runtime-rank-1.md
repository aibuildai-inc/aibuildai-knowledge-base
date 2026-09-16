# 1st Place Solution for the Google - Fast or Slow? Predict AI Model Runtime Competition

Competition: predict-ai-model-runtime
Rank: #1
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456343

First of all, thanks Kaggle and Google for hosting this great competition. Also, I'd like to thanks my teammates and friends @tomirol and @thanhhau097a, which made this journey even more special.

# Context
- Business context: https://www.kaggle.com/competitions/predict-ai-model-runtime/overview
- Data context: https://www.kaggle.com/competitions/predict-ai-model-runtime/data

# TLDR:
- We pruned and compressed the layout graphs in order to increase the efficiency of our experiments
- We removed duplicated configs for layout
- We changed the `node_feat` to use -1 padding instead of 0
- We used the provided `train`, `val`, `test` splits since we found good correlation with LB
- Data preprocessing:
    - StandardScaler for `node_feat[:134]`
    - Shared learned embedding (4 channels) for `node_feat[134:]` and `node_config_feat`
    - Features are concatenated before first linear
- Models with the following architecture:
    - Linear on the input features
    - 2x graph conv with attention blocks 
        - `InstanceNorm` -> `SAGEConv` -> `SelfChannelAttetion` -> `CrossConfigAttetion` -> `+residual` -> `GELU`
    - Global (graph) mean pooling
    - Linear logit layer
- `PairwiseHingeLoss` function

# Data preparation
We joined the competition with a bit less than a month to finish, so one of the first problems we tried to tackle was the  low efficiency in our training jobs.

## Graph pruning
For layout, we noticed that only `Convolution`, `Dot` and `Reshape` were configurable nodes. Also, in most cases, the majority of nodes would be identical across the config set. Thus, we opted for a very simple pruning strategy where, for each graph, we would only keep the nodes that were either configurable models themselves or were connected to a configurable node, i.e., input or output to a configurable node. By doing this, we would transform a single big graph into multiple (possibly disconnected) sub-graphs, which was not a problem since the network has a global graph pooling layer in the end that fuses the sub-graphs information. This simple trick reduced 4 times the vRAM usage and sped up training by a factor of 5 in some cases.

## Deduplication
Most of the configuration sets for layout contain a lot of duplication. However, the runtime for the duplicated configs can vary quite a bit and make training less stable. Thus, we opted for removing all the duplicated configs for layout.

## Compression
Even with pruning and de-duplication, the RAM usage to load all configs to memory for NLP collection was super high. We circumvent that issue by compressing `node_config_feat` beforehand and only decompressing it on-the-fly in the dataloader after config sampling. This enabled us to load all data to memory at the beginning of training, which reduced IO/CPU bottlenecks considerably and allowed us to train faster.

The idea behind the compression is that each `node_config_feat` 6-dim vector (input, output and kernel) can only have 7 possible values (-1, 0, 1, 2, 3, 4, 5) and, thus, can be represented by a single integer in base-7 (from 0 to 7^6).

## Changing pad value in `node_feat`
We noted that the features in `node_feat` were 0 padded. Whilst this is not a problem for most features, for others like `layout_minor_to_major_*` this can be ambiguous since 0 is a valid axis index. Also, the `node_config_feat` are -1 padded, which makes it incompatible with `layout_minor_to_major_*` from `node_feat`. With that in mind, we re-generated `node_feat` with -1 padded and this allowed us to use a single embedding matrix for both `node_feat[134:]` and `node_config_feat`.

# Data preprocessing
For layout, we split `node_feat` into `node_feat[:134]` and `node_feat[134:]` (`layout_minor_to_major_*`). The former was simply normalised using a `StandardScaler`, while the latter, along with `node_config_feat`, was fed into a learned embedding matrix (4 channels). We found that the normalisation is essential here since `node_feat` has features like `*_sum` and `*_product` that can be very high and, consequently, disrupt the optimisation.

For `node_opcode`, we also used a separate embedding layer with 16 channels. The input to the network is the concatenation of all features aforementioned and, for each graph, we sample on-the-fly 64 (default) or 128 (random) configs to form the input batch. For tile, on the other hand, we opt to use late fusion to integrate `config_feat` into the network.

# Network architecture
Our network architecture was quite simple. We first feed the input features to a Linear block to map it to a 256d embedding vector followed by 2x Conv blocks, global graph mean pooling and a final linear layer.

As for the graph convolutional layer itself, we tried many types but none was better than `SAGEConv`. In particular, I had good experience with GAT variants in the past but none worked well in this competition. If I were to guess the reason, I'd say that in the other application the graph itself was quite noisy, so attention helped to "ignore" the connections that were not meaningful. However, for TPU graphs, all connections are "real" and important so graph attention was not that helpful. Nonetheless, we found two other types of attention that were useful: self-channel attention and cross-config attention.

## Self-Channel Attention
We borrowed the idea from Squeeze-and-Excitation to create a channel-wise attention layer. We first apply a Linear layer to bottleneck the channel dimensions (8x reduction) followed by `ReLU`. Then, we applied a second linear layer to increase the channels again to the original value followed by sigmoid. We finish by applying element wise multiplication on the obtained feature map and the original input.

The idea behind this is to capture the correlations between channels and use it to suppress less useful ones while enhancing others.  

## Cross-Config Attention
Another dimension that we can exploit attention is the batch plane (cross-configs). We designed a very simple block that allows the model to explicitly "compare" each config against the others throughout the network. We found this to be much better than letting the model infer for each config individually and only compare them implicitly via the loss function (`PairwiseHingeLoss`). The attention code is as follows:
```
class CrossConfigAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x of shape (nb_configs, nb_nodes, nb_features)
        scores = (x / self.temperature).softmax(dim=0)
        x = x * scores
        return x
```

By applying this simple layer after the self-channel attention at every block of the network, it gave us a huge boost for default collections. For inference, we simply use a reasonably large batch size of 128. However, since the prediction depends on the batch, we can leverage it further by applying TTA to generate `N` (10) permutations of the configs and average the result after sorting it back to the original order.

## Linear/Conv blocks design
To create our Linear/Conv blocks we followed the good practices in computer vision. We start by using `InstanceNorm` to normalise the input feature map, followed by `Linear`/`SAGEConv` layer, `SelfChannelAttetion` and  `CrossConfigAttetion` (we concat the output with its input to preserve the individuality of each sample). Then, we sum the residual connection and finish with `GELU` and dropout.

# Ensembling
Our best single model prediction scored 0.714 (0.748) on private (public) LB. However, since the number of test samples is quite low for some collections, we opted to use ensembles to improve the results and prevent shaking up. Our best result was 0.736 (0.757) LB by using the simple average of 5-10 models for each collection but we sadly didn't select this sub.

# Things that didn't work
- Train together on both random and default data
- 2nd level stacking on top of model's embeddings/predictions (worked well locally but not that well on LB)
- Pseudo labelling test set
- Finetunning models for specific graphs in test set (worked well locally but not that well on LB)
- Different ways to sample the configs, e.g., annealing the runtime spacing between configs
- Other loss functions
- Gradient accumulation and training with more than one graph at a time

# Sources
- Our code can be found [in this GitHub repo](https://github.com/thanhhau097/google_fast_or_slow/tree/main)
- Our work was based/forked on @werus23 [public kernel](https://www.kaggle.com/code/werus23/tile-xla-end-to-end-train-infer) (thank you!)
- The modified layout dataset with -1 padding can be found [here](https://www.kaggle.com/datasets/tomirol/layout-npz-padding)

I made a quick diagram of our network, it's a bit crap but should give a good idea 😅


