# 6th solution: Node-level Instance Norm + Residual  SageConv on 5-hop-neighbour Subgraph

Competition: predict-ai-model-runtime
Rank: #6
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456084

# **Training and inference code can be downloaded from:**
https://github.com/hengck23/solution-predict-ai-model-runtime/  
&nbsp;


## 1. Layout runtime prediction
.png?generation=1701482437470415&alt=media)

main problem:
- we have very large graph as input. how to design learning model and algorithm that can fit into gpu memory? 

summary of approach :
- instead of using the whole graph, we can reduce it by considering only the 5-hop neighbours from node marked as "config id". We call this 5-hop-neighbour subgraph.  We think this is reasonable becuase since we are comparing relative ranking of 2 graphs,  and we just need to input the "difference nodes"  (instead of the whole graph) to the neural net.

.png?generation=1701483096930612&alt=media)

- using the reduced subgraph, i can sample 32 to 100 configurations using all full subgraphs with a single 48-GB  GPU card at training.
- batch size is not an issue here, bcuase i am using gradient accumulation. We accumuluate over one subgraph at a time when training a batch.

```
optimizer.zero_grad()

for b in range(batch_size):
        r = batch[r] 
        loss = net(r)  # forward one subgraph 
        scaler.scale(loss).backward() #backward accumuate gradient
 
scaler.step(optimizer) #update net parameters
scaler.update()
```
- normalisation is important. We use "graph instance norm" (over node), see paper[1], which works well with gradient accumulation 

- we use pairwise ranking loss in training loss.

- We try 2 GNN: 
   - 4-layer SAGE-conv[2] with residual shortcut
   - 4-layer GIN-conv[3] 

SAGE-conv is better than GIN-conv.

## 2. Tile runtime prediction
.png?generation=1701482450756660&alt=media)

main problem:
- There is no issues here as the graph in the kaggle training data are much smaller. These are actually subgraphs of the much larger original computation graph.

summary of approach
- We still use "graph instance norm" (over node) [1], and smae gradient accumulation apporach, with batch size =64.
- We try both SAGE-conv[2] and GAT-conv[4].  GAT-conv gives better results.
- Since we are interested in top 5 ranks, we find listMLE is a better loss.

---

## [Reference]

[1] GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training
https://arxiv.org/abs/2009.03294  
[2] Inductive Representation Learning on Large Graphs
https://arxiv.org/abs/1706.02216  
[3] How Powerful are Graph Neural Networks?
https://arxiv.org/pdf/1810.00826.pdf
[4] Graph Attention Networks 
https://arxiv.org/abs/1710.10903

---

## local validation and public/private score

The metric are: slowndown (top-5) for tile and kendall tau for layout.
.png?generation=1701488206301120&alt=media)

---

## Acknowledgment
## *"I would like to express my sincere gratitude to HP for the generous provision of the  Z8-G4 Data Science Workstation that was instrumental in the successful completion of kaggle competition. The two 48GB Nvidia Quadro RTX 8000 GPU cards give me a distinct advantage to easily bulid models with the largest public graph dataset TPUGraphs, with 100 millions graphs of 10 thousands nodes."*
