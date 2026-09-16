# Public 6th Private 14 Solution

Competition: open-problems-multimodal
Rank: #13
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/366392

## Intro
- I have been mainly working on the Cite part. I tried many things, 
- multi part by @paragkale https://www.kaggle.com/paragkale/private-14th-public-6th-multiome-portion
- The following tricks gave me the most gain.
- I will update in this post about the code and what didn't work.

## Extra Data
- The raw count data released by the host.

##Dimensionality Reduction

I think the most helpful one are:
- sklearn.decomposition.TruncatedSVD (128 comps)
- Self-made denosing auto encoder (128 hidden nodes)

## Direct Features
- Direct features based on matching the names
- Direct features based on absolute correlation to targets
- Direct features based on the list shared by the hosts in this thread https://www.kaggle.com/competitions/open-problems-multimodal/discussion/366392

## Use base models outcomes as NN inputs feature for ensembling

It is known that MSE is not a really good loss function for the competition metric. Therefore within each fold, we trained 4 base models and used their features for the NN input.
- sklearn.linear_model.Ridge
- sklearn.linear_model.MultiTaskElasticNet
- sklearn.kernel_ridge.KernelRidge
- sklearn.ensemble.HistGradientBoostingRegressor

We added heavy noise to their predictions to make sure the NN can learn from other features as well
```
        self.blender = torch.nn.Sequential(
            GaussianNoise(self.blend_noise),
            torch.nn.Linear(out_dim * 4, 128),
            torch.nn.LayerNorm(128),
            activation(),
            torch.nn.Dropout(self.blend_dropout),
        )
```

## GroupK Cross Validation on Target Clusters
The tricks in this section increased both public and private LB, but we cannot compare the CV because it is a CV scheme change. Luckily it is (relatively, I guess?) performing well on both public and private.

It is known that there are some subtle domain shifts between train, private and public test sets. However, the difficulty is that the shift is happening in at least 3 directions (donor, day, cell types). To create a hard but not too hard CV scheme, we find that clustering the target values performed very well on both of the public and private leaderboard.

Let's consider the CV scheme selection as a spectrum:
- The easiest CV scheme: Random K fold (Downside: not representative of the test set)
- The hard CV scheme: GroupKfold by day/donor (Downside: too few fold to train)
- The hardest CV scheme 1: Time series split (Downside: wasting the last day data)
- The hardest CV scheme 2: Excluding the 1 day or 1 donor completely from the training set (Downside: too hard/defensive)

Another reason of doing the clustering is that the day here is categorical, however in real life, time is continuous. GroupK CV by day is not that satisfying.

The first image shows the target kmeans result (colors)  visualized with the tsvd targets (points):


Next, you can see the target clusters capture the cell type differences:


And the shifts of day and donor are not that significant compared to the cell types in the context of target clustering:



## Regularization/Augmentation
The tricks in this section increased both CV and LB.

### Seed-bagging
I think most people have done this, we trained the same model a few more times with the different seeds for blending.

### Mixup Augmentation and Stochastic Weight Averaging
The training is done on roughly 3 stages
#### 1. Mix up augmentation stage
Since all features are numerical values, mixup worked well.
```
def mixup_augmentation(x: torch.Tensor, y: torch.Tensor, alpha: float = 5):
    lam = np.random.beta(alpha, alpha)
    rand_idx = torch.randperm(x.shape[0])
    mixed_x = lam * x + (1 - lam) * x[rand_idx, :]
    target_a, target_b = y, y[rand_idx]
    return mixed_x, target_a, target_b, lam
```
#### 2. Normal training stage
#### 3. SWA stage
https://pytorch.org/docs/stable/optim.html#putting-it-all-together
This is similar to seed-bagging, I am not sure if they are overlapping or if they have their benefits here.
