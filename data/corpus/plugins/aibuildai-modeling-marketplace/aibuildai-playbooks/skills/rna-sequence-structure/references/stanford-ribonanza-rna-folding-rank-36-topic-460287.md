# 36th solution : Cheap Transformers that can run on gaming laptops and does not require a BPP matrix

Competition: stanford-ribonanza-rna-folding
Rank: #36
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460287

First of all, I would like to thank all the participants in the competition. My solution is a cheap transformer that does not require bpp-matrix and can learn and predict with poor computational resources named "Cheap Transformers"

# Generalization to long sequences
My starting point is [Iafoss's amazing work](https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb) where the reactivities are predicted by simple transformers.

It works very well in cross validation in random fold splitting but gives poor performance when train and validation data is split by length of sequence like following. 

```python
short_indices = df[df['sequence'].str.len() < 177].index.tolist()
long_indices = df[df['sequence'].str.len() > 177].index.tolist()
```

Therefore, I need another architecture to generalize to long sequences.

# Cheap Transformer

This is an architecture of cheap transformer

er. It just cut a certain length sequence from the original sequence.

When regression, it cuts the sequence in multiple ways and ensemble the prediction results for each character. The ensemble is important. It improves cv-score at about 0.005.



This architecture uses short sequences, reduces training time, and requires less GPU memory during execution. Therefore, it runs fast enough even with the GPU of a gaming laptop.

# Final submission
I prepared multiple models with cut sequence lengths of 64, 96, and 128 and with various transformer's parameters and created an ensemble. Pseudo labeling was also performed, but it had little effect on LB. As you can see the ensemble contributed LB but its effect is not so big, therefore, using single model would be the most practival.



# Scores
| Method | Public | Private |
| --- | --- | --- |
| Iafoss's architecture | 0.15669 | 0.20204  |
| Cheap transformers (len=64) | 0.15635 | 0.15661  |
| Cheap transformers (len=96) | 0.15458 | 0.15445  |
| Cheap transformers (len=128) | 0.15518 | 0.15625  |
|Public best (ensemble with Iafoss's arch) | 0.14716 | 0.18222  |
| ensemble | 0.15271 |  0.15346 |
