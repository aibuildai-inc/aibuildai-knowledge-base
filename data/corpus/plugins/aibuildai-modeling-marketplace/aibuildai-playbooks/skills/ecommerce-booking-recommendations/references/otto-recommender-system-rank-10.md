# 10th Place Solution

Competition: otto-recommender-system
Rank: #10
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/382851

Thanks OTTO & Kaggle to host such interesting competition, my solution is straight forward, just candidate generation + reranker, a short summary as below.

###### Candidate Generation

- **re-visit** - all visited items
- **co-visit1** - based on the public notebook with 2 improvement. Thanks @radek1 and @cdeotte    
    1. candidate generation by weight decay, e.g. a session has N items(idx1 by the reverse ts) & each item has M pairs of co-visit item(idx2 by time weight or type weight), score is calculated as 1 / idx1 / idx2 for each item & candidate pair, then sum the score as candidate score
    2. co-visit features for rerank model - counts & probs pairwise item co-occurrences, index distance & time distance(consider order or not), etc.
- **co-visit2(buy2buy)** - only carts or orders
- **co-visit3(next item)** - only consider last item when generate candidate
- **sknn1** - simple sknn with 100 most similair & recent session 
- **sknn2** - a variant of sknn named as Sequence and Time Aware Neighborhood, reference https://arxiv.org/abs/1910.12781 for both sknn implementation

###### Re-Rank Model

- **aid features** - simple stat by time window(7, 14, 21)
- **session features** - length, aid count, count by type, time to the end of prediction period
- **revisit aid X session**, clicks/carts/orders count, absolute/relative position
- **co-visit features** - as mentioned before, generate a lot feature to describe pairwise item co-occurrences
- **sknn features** - only candidate score/rank
- **similarity features** - similarity between candidate item and session by item embedding of word2vec & implicit' s BPR

Finally, 3 models(clicks/carts/orders) is trained by LightGBM(lambdarank)
