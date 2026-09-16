# 9th place solution

Competition: playground-series-s6e3
Rank: #9
Source: https://www.kaggle.com/c/playground-series-s6e3/writeups/9th-place-solution

Firstly, a big thank you to the organizers and the community. My final ensemble was a blend of Gradient Boosting, Neural Networks, and Linear Models. I used two different feature sets for all the models and blended the predictions in the final.

**Final CV Score: 0.919911**

# Feature Engineering

For feature engineering, I relied on proven community techniques and public insights [from here](https://www.kaggle.com/code/blamerx/s6e3-ridge-xgb-n-gram-0-91927-cv), and [here](https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit), which provided a robust baseline for my models.

# Models

To reach the 9th position, I ensemble a massive library of models using the above-mentioned feature engineering techniques. My final solution included:

- GBDT Models: CatBoost (Plain, Ordered), XGBoost (with Pseudo-labeling, Target Encoding, and Residual-based versions), LightGBM (GOSS and TE), and HistGradientBoosting.

- Deep Learning for Tabular Data: A heavy emphasis on specialized architectures, including RealMLP, TabM, TabTransformer, FT-Transformer, ResNet, and TorchFrame.

- Specialized Architectures: I also experimented with and included GNNs, GateNet, DANet, DCNv2, Bartz and xLearn to ensure the ensemble had maximum architectural diversity.

- Refined Variants: Several models, including TabM, RealMLP, DANet, GateNet, DCNv2, and FTT, were trained from scratch with the idea from @yunsuxiaozi to squeeze out marginal gains in CV.

# Validation & Ensemble

- Cross-Validation: I used a strict Stratified K-Fold strategy.
- Ensemble Strategy: I converted all OOF predictions to ranks using scipy.stats.rankdata and used a GPU-accelerated Hill Climbing algorithm. 

| Model | OOF Score (AUC) |
| --- | --- |
| realmlp | 0.919389 |
| xgb with residuals | 0.919291 |
| realmlp_32 | 0.919285 |
| xgb with te | 0.919253 |
| realmlp_scratch | 0.919079 |
| lgbm_te | 0.919062 |
| cat_te | 0.919017 |
| tabm with te | 0.918979 |
| cat_ordered | 0.918919 |
| lgbm_te_goss | 0.918400 |
| danet_scratch | 0.918082 |
| tabm_scratch | 0.918018 |
| hgb_te | 0.918008 |
| xgb_pseudo | 0.917936 |
| tabm_pseudo | 0.917891 |
| cat | 0.917779 |
| ftt_scratch | 0.917532 |
| hgb | 0.917316 |
| tabm | 0.917243 |
| ftt | 0.917185 |
| torch_frame | 0.916991 |
| xlearn | 0.916930 |
| gatenet_scratch | 0.916844 |
| tabm_v2 | 0.916714 |
| resnet | 0.916700 |
| xgb | 0.916601 |
| bartz | 0.916583 |
| tab_transformer | 0.916480 |
| nn | 0.916369 |
| lr | 0.916020 |
| gnn | 0.915442 |
| dcnv2_scratch | 0.912211 |


# References

- [RealMLP](https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit)
- [RealMLP from scratch](https://www.kaggle.com/code/yunsuxiaozi/realmlp-from-scratchcv-0-91908)
- [LR and Torch NN](https://www.kaggle.com/code/cdeotte/chatgpt-vibe-coding-3xgpu-models-cv-0-9178)
- [GNN](https://www.kaggle.com/code/cdeotte/gnn-starter-cv-0-9155-with-hill-climbing-demo)
- [Bartz](https://www.kaggle.com/code/cdeotte/bartz-starter-cv-0-9164-with-hill-climbing-demo)
- [Trompt](https://www.kaggle.com/code/yekenot/ps-s6-e3-trompt-pytorch-frame)
- [TabTransformer](https://www.kaggle.com/code/include4eto/tabtransfomer-chatgpt-vibe-coding)

[Solution Link](https://github.com/mert-byrktr/KAGGLE-PS-S6E3-PREDICT-CUSTOMER-CHURN)
