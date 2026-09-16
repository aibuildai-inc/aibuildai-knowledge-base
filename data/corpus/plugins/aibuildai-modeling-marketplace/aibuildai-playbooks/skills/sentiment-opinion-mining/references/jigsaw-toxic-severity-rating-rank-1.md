# 1st place solution with code

Competition: jigsaw-toxic-severity-rating
Rank: #1
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306274

Public LB looks misleading so I focused on the validation performance only. Since it's unclear whether there is any distribution shift between validation data and private LB data, I only considered linear models build on validation data to maximize the robustness of the solution. To achieve this, I trained models on the Jigsaw2018 data and use the predicted probabilities (6 output) as input features and fit a linear model on the validation data. Weights were optimized with genetic algorithms. I did the same for Jigsaw2019 data (7 output). The Ruddit data have only a single label so no optimization is done on the validation data. 

The table below shows the performance of all my models. I only used roberta and deberta models as they gave the best performance. Final submission is a weighted rank average of 15 models. "-l" means I included the duplicates between Jigsaw18 and validation.

| model | data | validation | public LB | private LB |
| --- | --- | --- | --- |
| roberta-base | jigsaw18 | 0.7023 | 0.7815 | 0.8052 |
| roberta-large | jigsaw18 | 0.7035 | 0.7788 | 0.8064 |
| deberta-base | jigsaw18 | 0.7040 | 0.7598 | 0.8030 |
| deberta-large | jigsaw18 | 0.7050 | 0.7906 | 0.8139 |
| roberta-base-l | jigsaw18 | 0.7028 | 0.7690 | 0.8070 |
| roberta-large-l | jigsaw18 | 0.7027 | 0.7737 | 0.8013 |
| deberta-base-l | jigsaw18 | 0.7030 | 0.7474 | 0.8013 |
| deberta-large-l | jigsaw18 | 0.7044 | 0.7716 | 0.8085 |
| roberta-base | jigsaw19 | 0.7008 | 0.7617 | 0.8020 |
| roberta-large | jigsaw19 | 0.6991 | 0.7468 | 0.7968 |
| deberta-base | jigsaw19 | 0.7026 | 0.7403 | 0.7958 |
| roberta-base | ruddit | 0.6859 | 0.8108 | 0.7845 |
| roberta-large | ruddit | 0.6865 | 0.8132 | 0.7955 |
| deberta-base | ruddit | 0.6880 | 0.7903 | 0.7880 |
| deberta-large | ruddit | 0.6942 | 0.8296 | 0.7989 |
| ensemble | jigsaw18 |  | 0.7763 | 0.8103 |
| ensemble | jigsaw19 |  | 0.7509 | 0.8012 |
| ensemble | ruddit |  | 0.8235 | 0.7983 |
| ensemble | all15 |  | 0.7879 | 0.8139 |


At last I would like to thank Jigsaw and Kaggle for hosting this competition.

Link to inference code: https://www.kaggle.com/wowfattie/notebook9298460840
Link to training code: https://github.com/GuanshuoXu/Jigsaw-Rate-Severity-of-Toxic-Comments
