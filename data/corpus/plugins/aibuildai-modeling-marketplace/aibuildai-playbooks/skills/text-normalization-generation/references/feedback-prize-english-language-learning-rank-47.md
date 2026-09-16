# 47th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #47
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369770

First of all, thanks for hosting this competition and congratulations to all the participants. This is my first time to get medal in Kaggle, I've  learned a lot from this experience.

# Overview
My solution is training a lots of models with different parameters(pooling techniques, backbones, max_len, seed) and ensemble them by weighted average. My best CV score is 0.4484.

# Best Model Configuration
- Backbone: deberta-v3-base
- Pooling: mean
- Max_len: 512
- Learning rate:
  - Encoder: 2e-4 (layer-wise lr decay: 0.25)
  - Decoder: 1e-3
- Epoch: 4 (1 for warm up)
- Batch size: 8
- Fold: 5
- Seed: 42

# Parameters
- Backbone: deberta-v3-base, deberta-v3-large, deberta-v3-small
- Pooling: mean, cls, max, weighted layer(use last 4 or 5 layers)
- Max_len: 512, 768, 1024, 1536
- Seed: 13, 42
- Epoch: 3, 4, 5

# Ensemble
Find weights by [this](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/363773) in my models pool, filter out the model whos weight < 1e-3 and re-find weights.

# What worked
- [High Impact] Layer-wise Learning Rate Decay
- [low Impact] Add special token: "\n\n", "\r\n\r\n"
- [low Impact] First epoch warm up
- [low Impact] Freeze layers

# What Didn’t Work
- AWP
- Pseudo Labels
- Last Layer Re-initialization

# Important Citation
- [FB3 / Deberta-v3-base baseline [train]](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
- [Strategies For Ensembling Models Using OOF Predictions](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/363773)
- [Utilizing Transformer Representations Efficiently](https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently)

# Thanks and Acknowledgements
Thanks to competition organizers for hosting this competition and everyone who shared their knowledge during the competition. I am so happy to learn a lot from it!
