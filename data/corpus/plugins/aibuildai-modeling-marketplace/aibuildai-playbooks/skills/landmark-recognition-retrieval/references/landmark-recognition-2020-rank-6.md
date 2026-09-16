# 6th place solution

Competition: landmark-recognition-2020
Rank: #6
Source: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187961

We thank all organizers for this very exciting competition.  
Congratulations to all who finished the competition and to the winners.

## Summary

- kNN based on global descriptor
- Non-landmark filtering with similarity between test set and GLDv2 test set
- We tried local descriptor methods but they were not used in our final submission


## Model Details

We trained CosFace based global features models.  
The settings are almost the same as our models in the retrieval2020 (See https://www.kaggle.com/c/landmark-retrieval-2020/discussion/175472)

- Backbones: Ensemble of ResNeSt101, ResNeSt101, ResNeSt200
- Pooling: GeM (p=3) (Replace GeM p=3 with p=4 in testing)
- Head: FC->BN->L2
- Loss: CosFace with Label Smoothing
- Data Augmentation: HorizontalFlip, RandomResizedCrop, Rotation, RandomGrayScale, ColorJitter, GaussianNoise, Normalize, and GridMask
- LR: Cosine Annealing LR with warmup, training for 30 epochs + refine 5epochs
- Input image size in training: 352 (in refine: 640)
- Adding 600 non-landmark images sampled from the test set of the last year's recognition competition in training (this made the total 81,314 classes).

Additionally, we tried "Inplace Knowledge Distillation" by MutualNet paper (https://arxiv.org/pdf/1909.12978.pdf) to train multi-scale image features efficiently.  
It has been adopted for all models.


## HOW descriptor based ASMK similarity

We tried HOW descriptor (https://arxiv.org/pdf/2007.13172.pdf) as an alternative of the global feature model.  
HOW aggregates CNN based local descriptors into a single global descriptor with ASMK.  
We trained a ResNet50-based HOW descriptor model (our pytorch implementation) and got public score = 0.5284 (private = 0.5044).  
One of our final submission was based on blending between global cosine similarity and ASMK similarity, that achieved the best public score = 0.6298. However, the other global-feature-only approach achieved the better private score.

| Final submissions | Private | Public |
|-------------------|---------|--------|
| Global features   | 0.5983  | 0.6271 |
| Global + HOW ASMK | 0.5960  | 0.6298 |


## What did not work

- Reranking top-k with HOW ASMK similarity (Blending similarity was better)
- Reranking top-k with DELG local descriptors
- DBA
- Replacing the private train set with GLDv2clean
  - Undersampling up to 350 images per class
  - Removing classes that do not exist in the private train set
