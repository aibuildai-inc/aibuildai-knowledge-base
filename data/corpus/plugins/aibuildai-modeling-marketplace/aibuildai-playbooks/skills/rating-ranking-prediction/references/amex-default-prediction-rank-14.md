# 14th Place Gold – NN Transformer using LGBM Knowledge Distillation

Competition: amex-default-prediction
Rank: #14
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/347641

Thank you Amex for sharing your data and hosting a fun tabular competition. Thank you Kaggle. Thank you Kagglers for sharing many helpful discussions and notebooks. Thank you Raddar and Martin for your contributions.

# Solution Overview
My solution is a 50%/50% ensemble of LGBM and NN Transformer. The LGBM is based on Martin’s amazing public LGBM [here][1] and the NN Transformer is based on my public Transformer [here][2]. 

The secret sauce is how we train the Transformer. We first use knowledge distillation from our trained LGBM before fine tuning with the train targets. Furthermore, both train and test data are used for knowledge distillation which helps the Transformer learn the test data distribution.



# NN Transformer Training
My public notebook transformer has 2 layers with skip connections (to make training easy). When using knowledge distillation, we can train a deeper transformer successfully. My final solution uses a 4 layer transformer without skip connections. We also added a GRU layer after transformer blocks and before final classification layers.

Training is done using 4 cycles of cosine learning schedule. In the first cold start cosine cycle, we pretrain (i.e. Knowldege Distillation) the Transformer using concatenated rows of both LGBM OOF preds and LGBM test preds and leave probabilities between 0 and 1 (i.e. soft labels). During the second cosine cycle, we use a warm start, reduce the learning rate and train with the hard (0 or 1) train targets. For the third and fourth cycle, we repeat cycles one and two.



# Model Performance
When creating our submission.csv file from our two models, we can use the normal **K-Fold** LGBM OOF preds and normal LGBM test preds. So making a submission is fast an easy. Additionally, we average 5 seeds per model (and slight model variations) for improved performance.

To tune our two models and compute optimal hyperparameters, we need a leak free reliable CV score. Leak-free CV score is created using **Nested K-Fold** CV. We divide each of 10 outer folds into 10 inner folds. We then train 100 models using GBT. Then each 1 of 10 outer folds has its unique OOF preds and unique test preds. These individualized OOF and test preds are created using only train targets from within the corresponding outer fold train data.

When computing leak free CV score, we find that our NN Transformer has **CV 0.798 / LB 0.799**, our LGBM has **CV 0.799 / LB 0.799** and our 50%/50% ensemble has **CV 0.800 / LB 0.801**.

# Fast Experimentation
Fast experimention was done using GPU. Thank you [Nvidia][5] for providing me compute resources for this competition. Experiments were accelerated using 4xV100 32GB.

Feature engineering was explored using [RAPIDS cuDF][3] which performs operations like dataframe groupby aggregation on GPU 10-100x faster than using CPU. Many GBT experimental models were trained and evaluated using fast GPU XGB. With 1xV100 GPU, XGB can train 100 models for Nested 10 in 10 K-Fold (i.e. 100 models) on full data in only 2 hours. 

Feature selection was performed using both XGB feature importance and permutation importance. Using [RAPIDS FIL][4], we can perform permutation importance where we randomly shuffle each feature column 10 times for each of 10 folds (and average 100 results) in blazing speed! 

Each of 1000s of feature columns, we infer 100 times. This is a total of 100,000s of model inferences where each model is 1000s of individual trees! Using [RAPIDs FIL][4], we can perform this quickly! Note that we can even take an existing CPU LGBM Dart model and convert it into a GPU [RAPIDS FIL][4] inference model and perform permutation importance on existing LGBM Dart Models in blazing speed!

[1]: https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977
[2]: https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-790
[3]: https://rapids.ai/
[4]: https://docs.rapids.ai/api/cuml/stable/api.html#forest-inferencing
[5]: https://www.nvidia.com/en-us/
