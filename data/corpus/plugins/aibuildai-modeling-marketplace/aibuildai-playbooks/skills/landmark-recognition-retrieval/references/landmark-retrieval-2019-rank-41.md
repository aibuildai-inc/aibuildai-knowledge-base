# Our solution: 0.10439 private 0.08832 public (code available)

Competition: landmark-retrieval-2019
Rank: #41
Source: https://www.kaggle.com/c/landmark-retrieval-2019/discussion/94386#latest-543711

Our solution consists of two steps, either single model (our best single model achieves 0.06335 public 0.08813 private), and ensembling multiple models which achieves 0.10439 private 0.08832 public.

## [Single model CNN extraction + NN search](https://github.com/antorsae/landmark-retrieval-2019/blob/master/submission-trained.ipynb)

This notebook extracts the last convolution layer of a given architecture applied GeM pooling and performs L2 normalization.

It uses augmentation (images are LR flipped) so both the index and queries features
are duplicated.

Once features are extracted, it performs regular nearest neighbour search using PCA, whitening and L2 normalization
and the resuls are ready for a submission.

It also saves the results of nearest neighbour search for ensembling.

## [Ensembling](https://github.com/antorsae/landmark-retrieval-2019/blob/master/ensemble.ipynb)

This notebook takes the results of nearest neighbours search for each query (and flipped LR queries)
and aggregates distances of different runs (different architectures) and builds a submission accordingly.

For flipped LR images, we pick the minimum distance.

## Some results

[image]
[image]
[image]

All code here: https://github.com/antorsae/landmark-retrieval-2019
