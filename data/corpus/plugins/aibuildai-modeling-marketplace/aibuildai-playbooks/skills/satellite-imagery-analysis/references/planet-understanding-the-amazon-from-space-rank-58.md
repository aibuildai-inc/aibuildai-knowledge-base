# 0.93017 solution

Competition: planet-understanding-the-amazon-from-space
Rank: #58
Source: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36938

I've just pushed pytorch code https://github.com/EdwardTyantov/pytorch-kaggle-amazon-space
Nothing special inside, but code is very clean &amp; structured, so it can help someone.

**Results**:

 * Best single model: mixnet_v6: public: 0.92905, private: 0.93071
 * Best blended ensemble: public: 0.93015, private: 0.93217
 * Best submit during competition: 0.93023: 0.93168 (also ensemble)

**Quick overview of the solution**:

Best single model (mixnetv6) solution consist of following tricks:

 * 6-channel input (3-jpg channel, NIR-chanell, NDWI-index, SAVI-index)
 * model: resnet18 on jpg, resnet 18 on nir+indexes, concat -&gt; 256 embedding FC + final FC
   * jpg branch lr modifier 0.05 to base LR, for nir branch - layer{3,4} - 1.0, layer{2,3} - 0.1, FC - 1.0 
 * plateau scheduller on val loss (cross-entropy), patience=3
 * early stopping: 6 epochs
 * train time augmentation: shift, flip, scale, rotate, transpose
 * test time augmentation, 6x: as-is, rotate 90*{1,2,3}, flip x, flip y
 * standart for this challenge searching thresholds for F2 (firstly I implement per class search - it is more consistent, but default on scale is better)
 
Best Ensemble:

 * trained various models on jpg, mix channels
   * models: 
     * densenet{121,169} on jpg, 5 folds
     * mixnetv6, mixnetv3 (different LRs) 5,6,7 folds
     * wideresnet on 6-channel (for mix branch unpretrained WideResNet), 7 folds
     * resnet18 + embedding FC on 8 folds
 * best submit was based on weighting predictions using holdout F2 score, weight=((score - min_score)/max_score)**0.5
