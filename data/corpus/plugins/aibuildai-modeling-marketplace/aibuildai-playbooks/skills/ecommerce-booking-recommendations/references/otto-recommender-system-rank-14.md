# 16th Place Solution (Association x UserCF x NN-based x Matrix Factorization x Covisit x LightGBM)

Competition: otto-recommender-system
Rank: #14
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/383382

Thanks to Otto and Kaggle team for hosting such a challenging competition. I tried almost all methods I could come up with and seeing the score improve step by step was thrilling, especially at the end of the competition.

The sad part is the Grandmaster/Master are allegedly involved in cheating, and many participants have been suffering/discouraged from such an unforgivable activity that hurts Kaggle's integrity and reputation.

I would like to share my solution, **which is 0.01% behind the Gold medal zone at this moment lol.**
Hope this helps, and I'm happy to answer questions.

# **Candidate Selection**
- revisit
- top20 covisitation matrix (two different matrix)
- top3 3-hop graph from covisitation matrix
  - lets say sessions's last aid is X, and X's top 3 covisit aids are A,B,C. I call them '1-hop' from X. A's top3 covisit aids are D,E,F. I call them '2-hop from X'. I did this for three hops so you have 3^3 cand to 1 aid)
- top10 from 3D-covisitation matrix
- top20 from simple itemcf/usercf (similaripy)
- top20 from gru4rec (recbole)

**The average cand per session is almost 100, and recall for valid is 0.6396**

# **Feature Engineering**
- simple interaction between aid and session (ex: time lapse from session's last action to candidate aid)
- session-only based (ex: num of clicks of that session)
- aid-only based (ex: num of clicks of that aid)
- covisitation matrix score from four different covisitation matrices
- association rules score (jaccard/dice/lift---)
- simple user/item cf score (using **'similaripy'**) 
- item2vec cosine similarity (using **'gensim'**)
- matrix factorization type similarity (using **'implicit'**)
  - BPR
  - ALS
  - LightFM
  - SVD 
- neural net based scores (using **'recbole'**)
 - lightGCN
 - GRU4Rec
 - BERT4Rec
 - RecVAE
 - SasREC
 - SRGNN

The total num of features is 291.
From my observation, there was no silver bullet feature(s). All types of features slightly contributed to the model. 
NN features are relatively costly because you need time and computing resources to get them but still get marginal boost.
Association rules and simple userCF are easy to handle because they are not so computationally intensive.

**Single model best CV is .5808 and public .60206/private .6017**

# **Modeling**
LightGBM Ranker with custom metrics(Recall at 20)
LightGBM binary/Catboost binary/Catboost Yetirank are all worse than LightGBM Ranker.

The final submission is the ensemble from five slightly different candidates/model specifications. Ensemble helped me a little(+0.03%).

# **What I should have done to get solo gold**
 - Catboost Pairwise (some report it works well)
 - More Candidate (maybe 150-200 per candidates)
