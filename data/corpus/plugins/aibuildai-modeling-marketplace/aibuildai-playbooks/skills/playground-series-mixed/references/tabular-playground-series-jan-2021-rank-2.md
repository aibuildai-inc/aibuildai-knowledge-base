# 2nd solution write up.

Competition: tabular-playground-series-jan-2021
Rank: #2
Source: https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216070

Sorry that I don't have a lot of time to write a long and detailed solution post. As a working dad of two kiddles under the pandemic, alone time is very precious.

So my approach to this competition is two-part, a boring part, and an interesting part. 

The boring part includes tweaking my old competition code to tune and train whatever model I am familiar with and save the out of folds for stacking. Since this data set on the surface requires little processing, I did the code in 2 hours and just have one spare machine at home running it for the entire month. It spits out 10 GBT models(3 lightgbm, 4 xgboost, 3 NGBoost), 2 SVM, 1 KNN, 2 Ridge, 1 Random Forest, 1 regularized greedy forest in the end. 

The interesting part, I guess is the search for NNs that can work close to what trees are. I did this mostly part during weekends. I tried out DAE briefly, but not able to get it working. So I used the swap noise part for training the supervised NN, hoping it could at least encourage the network a bit and it did helped. Here is some snippets of code on this:

```
def add_swap_noise_torch(X, ratio=.15, col_to_apply=[], return_mask=False):
    obfuscation_mask = torch.bernoulli(ratio * torch.ones(X.shape)).to(X.device)
    if col_to_apply:
        column_mask = torch.zeros(X.shape).to(X.device)
        column_mask[:, col_to_apply] = 1
        obfuscation_mask *= column_mask.float()
    obfuscated_X = torch.where(obfuscation_mask == 1, X[torch.randperm(X.shape[0])], X)
    if return_mask:
        return obfuscated_X, obfuscation_mask
    return obfuscated_X
```
So basically, I used this function to add noise to each mini-batch. With this noise and some tweaking on the structure of MLP, I got NN down to .702 CV as I noted [here](https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/208429#1161897). At this time, I was pretty sure DAE did the trick for @springmanndaniel but I don't think I have enough time and energy to really get it working. I have a feeling that each input variable is a composite of multiple components if I can break them into smaller parts in a meaningful way I could help the NN further. So I found myself the `dwarf` version of DAE that is this [RandomTreesEmbedding](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html) and I also [shared this too](https://www.kaggle.com/ryanzhang/ridge-on-random-tree-features). My NNs trained on this feature horizontally stacked to the original inputs, with swap noise applied to only the original inputs, I got my NNs down to .698 CV. And since. I trained about 6 NNs with slight differences in the end. 

My final submission is a ridge regression stacking all the above. And I am not going to share more code, because an exhausted dad's code at 2 AM is not readable.
