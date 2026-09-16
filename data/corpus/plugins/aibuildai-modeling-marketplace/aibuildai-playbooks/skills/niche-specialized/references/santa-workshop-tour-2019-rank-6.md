# Our trick

Competition: santa-workshop-tour-2019
Rank: #6
Source: https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/126255

1) Use LP 67xyz LP solution
2) Core trick in that LP is that `sum_i M_d[i][j] == sum_k M_{d+1}[j][k]` (which is not needed but pushes LP higher)
3) Our trick for easier cutting and branching is to have `sum_i M_d[i][j]` as separate variable, so solver can branch on it.

Side note:
We spend too much time on formulation with 175 variables per day and having 175*175 constrains. We had tricks like convex hulls, lazy constraints, ... but that was not that great.
