# 2nd Public/13th Private Solution

Competition: leash-BELKA
Rank: #13
Source: https://www.kaggle.com/c/leash-BELKA/discussion/519133

We would like to thank Kaggle for organizing such an interesting competition. We also appreciate @tetsuya3510 , @hengck23 , and @ahmedelfazouan for sharing notebooks that influenced our solution. A big thanks to my teammates @yyyu54 , @Ogurtsov, and @antoninadolgorukova for fighting side by side until we used up all 480 submissions. We all reached the Competitions Master at the same time!
# Approach Overview
We used separate approaches for molecules with shared building blocks and non-shared building blocks.
## 1. Shared Building Blocks
We used an ensemble of CNN, GBDT, and GNN models. 
### CNN models
It’s two variations of the great [public notebook](https://www.kaggle.com/code/ahmedelfazouan/belka-1dcnn-starter-with-all-data) by @ahmedelfazouan. 
**Data:** The same dataset that was used in the public notebook ([Link](https://www.kaggle.com/datasets/ahmedelfazouan/belka-enc-dataset)). 
**Model architecture:**

**The major changes:** We doubled the filter sizes from 32, 64, 96, to 64, 128, 192; kernel sizes were increased from 3, 3, 3, to 19, 9, 3. The ReLU activations were replaced by SiLU. Moreover, for the second model, we added a bidirectional GRU layer after the embedding and concatenated it with the convolution layers after global max pooling. 
**Training parameters:** See the table at the end.
We made a weighted average of the above two models trained with and without the validation set, using a total of four CNN models.
## Other models:
### XGBoost (Written in R), LightGBM (Written in Python):
To achieve maximum diversity, the models were trained with different features on different subsets of the train data. All models were trained separately for each protein([Code Link](https://www.kaggle.com/code/antoninadolgorukova/belka-gbdt-models-a-part-of-13th-place-solution)).
**Data:** A sample with all binding molecules and a random sample of non-binding molecules (50M or 40M in total for GBDTs and 10M for chemprop )
**Features:** For one model we added predictions from chemprop (version 2.0, the output of the 3rd linear layer in the FFN) to SECFP4 (bits=1024), and for two we added BB activity features (the fraction of compounds that bind when a given BB smiles occurs at a given position) to ECFP4 (bits=1024). For LightGBM we used SECFP4 (bits=1024) and SECFP6 (bits=2048).
**Training:** One model was trained five times on 5 parts of the train data, each one without 20% of random BBs and others on the 50M sample (excluding validation and test subsets). 
**XGBoost training parameters:** eta 0.05, max_depth: 25, subsample: 0.2, sampling_method: gradient_based, colsample_bytree: 0.4, min_child_weight: 4, gamma: 2, num_boost_round = 5000, early_stopping_rounds = 30.
**LightGBM training parameters:** max_depth: 11, bagging_fraction: 0.9, learning_rate: 0.05, colsample_bytree: 1, colsample_bynode: 0.5, lambda_l1: 1, lambda_l2: 1.5, num_leaves: 490, min_data_in_leaf': 50.
### GNN:
We used this public notebook by @hengck23 with minor changes (atom types list was truncated to actual atoms in train/test sets molecules).


For this part, we used weighted average to ensemble the predictions for each protein separately, using local scores (perfect correlation with LB): BRD4: 4 models, HSA: 7 models, sEH: 5 models.
## 2. Non-shared Building Blocks
Creating a reliable cross-validation for the molecules with nonshared BBs was difficult, so we conducted two ensemble methods based on the public score, and used them in the final submissions.
### Final submission 1 (public 0.488/private 0.275): Ranking ensemble 
For this solution, in order to minimize the fluctuations due to the differences between the public and private scores, we performed an ensemble of the following two ChemBERTa-based models, which gave relatively good predictions for all proteins in the non-share block.We converted the predictions of each model into ranks to account for differences in scale between models. 
**Model architecture:**


**Training parameters:** See the table below.
For the two models mentioned above, predictions were made for each epoch over five random folds. The average of these predictions was calculated over a total of 5 epochs × 5 folds × 2 models.

### Final submission 2 (public 0.529/private 0.277): Ranking ensemble by protein
We used one XGBoost model (ECFP4 features, trained as above on 5 subsets of train data, but with validation on subsets with non-shared BBs), and seven ChemBERTa models with different classification heads and training parameters. We scored each model for each protein, and considered the scores and diversity of predictions to select the weights (BRD4: 4 models, HSA: 3 models, sEH: 4 models). It is difficult to describe all eight models in detail, so the two models with the highest weights are described in the final submission 1 (feel free to ask anything!).

### Training parameters:

| Settings               | CNN                        | ChemBERTa                 |
| ---------------------- | -------------------------- | ------------------------- |
| Optimizer              | Adam                       | Adam                      |
| Learning rate          | 1e-3                       | 3e-4                      |
| Optimizer momentum     | beta1, beta2 = 0.9, 0.999  | beta1, beta2 = 0.9, 0.999 |
| Optimizer weight decay | 0.05                       | 0.01                      |
| Batch size             | 4096                       | 1024                      |
| Training epochs        | 50                         | 5                         |
| ReduceLROnPlateau      | patience, factor = 3, 0.05 | /                         |
| EarlyStopping          | patience = 5               | /                         |
