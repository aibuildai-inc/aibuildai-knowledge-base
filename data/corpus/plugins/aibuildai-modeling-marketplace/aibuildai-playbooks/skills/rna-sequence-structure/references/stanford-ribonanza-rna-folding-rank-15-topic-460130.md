# 15th place solution

Competition: stanford-ribonanza-rna-folding
Rank: #15
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460130

First and foremost, I'd like to express my gratitude to Kaggle for hosting this competition, and to the organizers for their active involvement and responsiveness in the forums. I gained valuable insights from participating in this competition. Special thanks to my teammates, Anton @ant0nch and María @manaves, and a shout-out to María for her extensive domain knowledge as a biotechnologist. Lastly, I want to extend my thanks to my employer, Freepik, for providing additional computing resources that proved crucial in the final weeks.

Our solution is based on a single transformer encoder module. We used the standard pytorch implementation with `d_model: 256`, `dim_feedforward: 768`, `num_layers: 16`, `dropout: 0.1`, GELU activation and normalization first.

The model was trained on a 5-fold split grouped by clusters, and clusters were obtained using KMeans of A, C, G, U -> 1, 2, 3, 4 mapped vectors.

We used AdamW with weight decay 1e-2 and OneCycleLR schedule with `pct_start: 0.02` and `max_lr: 1e-3`.

The encoder input is:

- A 252 dimensional vector encoding the sequence as the sum of nucleotide embeddings:

```
    self.embs = nn.Embedding(5, d_model - extra_features)
```

- We also encoded the secondary structures from the 47 provided algorithms using an embedding bag where only present data counted towards the mean. Data from the structures were encoded like this: . -> 1, ( -> 2, ) -> 3, [ or < or { -> 4, ] or > or } -> 5.

```
    self.se_embs = nn.EmbeddingBag(6, d_model - extra_features, padding_idx=0)
```

- The two embeddings were added together, and then the following features from the BPP files were concatenated for each nt: max, median, mean and std.

- Additionally, both the eternafold BPP data and adjacency matrices computed from the secondary structures were used as attention biases via a learnable linear layer. We used 48 matrices for regular pair adjacency and another 48 matrices for pseudoknot adjacency, plus the BPP matrix (so total = 48 * 2 + 1). We augmented the given data by generating additional secondary structures with mxfold2.

```
    self.algo_conv = nn.Conv2d(48 * 2 + 1, nhead, kernel_size=1, padding=0, bias=True)
```

- Positional encoding: we used the standard sinusoidal positional encoding from the original transformer paper. We attempt to achieve longer sequence generalization by assigning each nucleotide in the sequence random correlative positions sampled uniformly from `[0, 512[`. This hopefully forces the network to learn from the relative order of the nucleotides rather than their absolute position.

```
    # Make a sorted list of positions randomly sampled from [0, max_seq_len)
    rnd_pos = torch.empty((n, s), dtype=torch.int64, device=pred.device)
    for row in rnd_pos:
        row[:] = torch.randperm(self.max_seq_len, device=pred.device)[:s].sort().values
    pos = self.pos(rnd_pos) # N, S, E-4
```

We trained several models for 100 epochs with a batch size of 32 and different ablations of the above features, different SNR filtering strategies (>1 and >0.5) and different GroupKFold shuffle seeds. In the end, we averaged the predictions of all 5 folds of our 7 best models.

## What didn't work:

- training in bfloat16 gave substantially poorer results
- pseudolabels, or we just didn't know how to implement this properly
