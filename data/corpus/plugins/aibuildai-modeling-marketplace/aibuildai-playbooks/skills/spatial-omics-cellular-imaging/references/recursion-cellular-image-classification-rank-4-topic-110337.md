# 4th place solution

Competition: recursion-cellular-image-classification
Rank: #4
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110337

Congrats to all prize and medal winners!
Our brief solution summary:

1) heavy model ensemble
  - 1108-way classification
  - seresnext50, 101, densenet, efficientnet x 6C6, 5C6 4C6 channel selection x cross validation
  - 512x512 input, 90rot+clip aug

2) solve linear sum assignment problem to make the most of 'the groups of 277 per plate restriction'
https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/102905#latest-624588

```
from scipy.optimize import linear_sum_assignment
plate_prob = plate_prob / plate_prob.sum(axis=0, keepdims=True)
row_ind, col_ind = linear_sum_assignment(1 - plate_prob)
```

3) make psuedo label and go back to 1 (with a few selected models like efficientnet)

The code can be found in https://github.com/ngxbac/Kaggle-Recursion-Cellular
(mainly the 1) part)
