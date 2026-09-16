# 16th place solution

Competition: predict-ai-model-runtime
Rank: #16
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456489

Thanks for hosting the competition on a very interesting topic and congratulations to all the winners!

I'll share my solution briefly.

# Summary

- 3-hop subgraphs from configurable nodes
- Drop duplicated configs
- Listwise loss. In my experiments, listwise loss converged faster and performed better than pairwise loss.
- 3-layer `SAGEConv` with `LayerNorm` and residual connections. I implemented residual connections by simply adding the initial embedding to the output of each layer. Below is a code snippet.

```python
def forward(self, batch):
    node_opcode = batch.node_opcode.long()

    opcode_embeds = self.opcode_embedding(node_opcode)

    x = torch.concat([batch.node_feat, opcode_embeds, batch.node_config_feat * self.node_config_weights], dim=1)
    x = self.lin1(x)
    x = self.norm1(x).relu()

    x_init = x
    for i in range(self.n_layers):
        x = self.convs[i](x, batch.edge_index)
        x = self.norms[i](x).relu()
        x = x_init + x

    x = torch.concat([global_mean_pool(x, batch.batch), global_max_pool(x, batch.batch)], dim=1)
    x = self.dropout(x)
    x = self.readout(x)

    return x
```

- Models were trained separately for diffrent subtypes
- CV scores (provided train, valid splits)
    - xla default: 0.37
    - xla random: 0.71
    - nlp default: 0.55
    - nlp random: 0.96
    - tile: 0.97
    
    With these CV scores, I got a score of 0.684 (public) and 0.688 (private).
