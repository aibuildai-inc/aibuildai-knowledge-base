# Some Notes

Competition: asl-fingerspelling
Rank: #46
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/435231

Many more skilled participants have shared very rich and impressive solutions. I'm only documenting a few experimental insights to avoid adding something redundant to the discussion.

# LB History
| | public LB | private LB|
| ---  | --- | --- |
| max_len=188 | 0.770| 0.741|
| max_len=320 | 0.776| 0.744|
| Post Process by Chris Deotte | 0.778| 0.747|
| expand ratio 2->4 | 0.779| 0.752|
| epochs 126->200 | 0.780| 0.752|

# What Didn't Work
1. I attempted various normalization techniques, including global normalization, local normalization, and normalization centered around the nose. However, I'm unsure if there were any experimental errors, as I found that these approaches merely expedited the training convergence without improving the leaderboard performance. I'm uncertain if my conclusion is accurate.

2. Lowering the dropout of the final layer from 0.4 to 0.1 and removing dropout from all other layers would result in a decrease of 0.001 points in both the public test set and the private test set.

3. To place the batch normalization (BN) before the depthwise convolution (DW Conv) instead of after would result in a decrease of 0.001 points for both the public LB (Leaderboard) and private LB (Leaderboard).

4. Spatial Mask. It may lead to training instability when used. If other aspects are handled well (such as normalization), it might be usable.
