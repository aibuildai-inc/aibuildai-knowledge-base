# 33rd place solution: pre- and postprocessing

Competition: coleridgeinitiative-show-us-the-data
Rank: #33
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248463

I only had a couple of days to look at this problem, so the main insights I got were:
1) Pretty much every dataset string transforms into a chain of proper nouns after POS-tagging.
2) There is a limited amount of datasets both in train and test parts.

The idea was that even a weak model could show generalization ability if frequency-based filtering is applied after training on 'PROPN' tags instead of original strings. Overall, my pipeline consists of four parts:
1) replacing every proper noun chain with a single token. "Entity typing on the Open Entity dataset (Choi et al., 2018)" becomes "Entity typing on the PROPN dataset (PROPN, 2018)".
2) training a simple token classification model (most transformer-based models will do).
3) collecting top document candidates by using an output probability threshold.
4) decoding all collected 'PROPN' tokens to original strings and filtering them out by frequency.

Based on 4) implementation, this algorithm is capable of achieving a ~0.4 score on the private leaderboard (and frequently finds dataset mentions that look legit but are missing from annotations).
