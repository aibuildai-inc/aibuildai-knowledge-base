# 20th Place Solution

Competition: otto-recommender-system
Rank: #20
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/382771

# Acknowledgements
First, I would like to thank the organizers and those who shared knowledge. Especially I would like to thank @cdeotte and @radek1 for posting so many codes and discussions. I learned a lot from you. 

Considering the controversy about the possible cheaters, I will publish the entire code after everything is finalized and I will just share the ideas in this solution. (Although [Will says sharing code is not something we should withhold](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382440#2123460))


<br>
# Solution Summary


My code is here: https://github.com/kiccho1101/kaggle-otto2

<br>
# CV Strategy

- I used Radek’s CV strategy
- Although CV data was good for the local experiments, it took quite a long time to run experiments so I used only 1/20 sessions by random sampling. As a result, the average time that takes to run 1 experiment became about 10~30mins which allowed me to run many experiments quickly.
- Even if I sampled the data, the CV-LB score correlation was very stable.
- The data used for each stage is below

| Step                       | CV                     | LB                                              |
|----------------------------|------------------------|-------------------------------------------------|
| Candidate Generation       | TrainData + ValidDataA | TrainData + ValidDataA + ValidDataB + TestDataA |
| User Feature Creation      | ValidDataA             | TestDataA                                       |
| Item Feature Creation      | TrainData + ValidDataA | TrainData + ValidDataA + ValidDataB + TestDataA |
| User-Item Feature Creation | ValidDataA             | TestDataA                                       |
| Re-Ranking                 | ValidDataA             | TestDataA                                       |

<br>
# 1st Stage - Candidate Generation

<br>
### word2vec
- Trained word2vec model with aid sequences
- Retrieve top-k aids by word2vec embeddings (Used faiss-gpu to speed up)

<br>
### CoVis
- I used Chris’s co-visitation matrix to generate candidates

<br>
### Item MF (Item Matrix Factorization)
```
class ItemMFModel(nn.Module):
    def __init__(self, n_aid: int, n_factors: int):
        super().__init__()
        self.criterion = BPRLoss()
        self.n_factors = n_factors
        self.n_aid = n_aid
        self.aid_embeddings = nn.Embedding(self.n_aid, self.n_factors)

        initrange = 1.0 / self.n_factors
        nn.init.uniform_(self.aid_embeddings.weight.data, -initrange, initrange)

    def forward(self, aid_x, aid_y):
        aid_x = self.aid_embeddings(aid_x)
        aid_y = self.aid_embeddings(aid_y)
        return (aid_x * aid_y).sum(dim=1)

    def calc_loss(self, aid_x, aid_y, size_x, size_y):
        rand_idx = torch.randperm(aid_y.size(0))
        output_pos = self.forward(aid_x, aid_y)
        output_neg = self.forward(aid_x, aid_y[rand_idx])
        loss = self.criterion(output_pos, output_neg)
        return loss
```
- Trained aid_embeddings with BPR loss to make co-occurring embeddings become similar
- What worked
    - Multiply the inverse of item_size by loss (Removing popularity bias)
    - Multiply the inverse of ts_diff by loss (The closer the co-occur timing is, the more similar the embeddings become)

<br>
### User MF (User Matrix Factorization)
```
class UserMFModel(nn.Module):
    def __init__(self, n_session: int, n_aid: int, n_factors: int):
        super().__init__()
        self.n_factors = n_factors
        self.n_session = n_session
        self.n_aid = n_aid

        self.session_embeddings = nn.Embedding(self.n_session, self.n_factors)
        self.aid_embeddings = nn.Embedding(self.n_aid, self.n_factors)

        self.criterion = BPRLoss()

        initrange = 1.0 / self.n_factors
        nn.init.uniform_(self.session_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.aid_embeddings.weight.data, -initrange, initrange)

    def forward(self, session, aid, aid_size):
        session_emb = self.session_embeddings(session)
        aid_emb = self.aid_embeddings(aid)
        return (session_emb * aid_emb).sum(dim=1)

    def calc_loss(self, session, aid, aid_size):
        rand_idx = torch.randperm(aid.size(0))
        output_pos = self.forward(session, aid)
        output_neg = self.forward(session, aid[rand_idx])
        loss = self.criterion(output_pos, output_neg)
        return loss
```
- Trained session_embeddings and aid_embeddings with BPR loss
- What worked
    - Multiply the inverse of item_size by loss (Removing popularity bias)
    - Multiply the inverse of ts_diff by loss (The closer the co-occur timing is, the more similar the embeddings become)

<br>
### Item CF
- Implemented item cf with polars
- Calculated the similarity weights for each item-item pair and retrieved candidates by getting the most similar items based on sum/min/max/mean of weights
- What worked
    - Multiply the inverse of item_size by weight (Removing popularity bias)
    - Multiply the inverse of ts_diff by weight (The closer the co-occur timing is, the bigger the weight becomes)
    - Multiply trend coefficient (The more ts is recent, the bigger the weight becomes)

<br>
### User CF
- Implemented in the same way as item cf

<br>
# 2nd Stage - Re-Ranking
- Feature
    - Created ~200 features in total
    - pl.col(’ts’).agg([mean,min,max,std]).over({’session’ or ‘aid’})
    - candidate_selected_(count, rank)
    - candidate_(score, rank, selected)
    - (inter, click, cart, order)_hour_mean.over({’session’ or ‘aid’})
    - (inter, click, cart, order)_count.over({’session’ or ‘aid’})@(3d, 7d, 14d, 21d)
    - aid_multi_(inter,click,cart,order)_prob
- Model
    - LightGBM Ranker (lambdarank)
    - CatBoost Classifier (Logloss)
    - CatBoost Ranker (YetiRank)

<br>
# What did not work well
- Transformers
- GRU
- CDAE
- RecVAE
- Implicit(ALS, BPR)
- Clustering by item embeddings
- Popular items
- Stacking
- Pseudo Labeling

<br>
# Lessons Learned
- polars is all you need to create features
    - At first, I created features with pandas and cudf but I switched to polars during the competition because polars is fast, memory efficient, and the syntax is easy to understand.
- Creating a good baseline is very important
    - It took me 1~2 month to create a baseline pipeline
    - Thanks to Chris’s great discussion post, I was able to create a 2-stage baseline from scratch (I will publish the code on GitHub soon)
- Feature Store is useful
    - I stored intermediate files to parquet files, which saved me to run experiments quickly in a more reproducible way.

<br>
# Score Timeline
| CV (1/20 sampled) | CV    | Public LB | Description                   |
|-------------------|-------|-----------|-------------------------------|
| 0.536             | 0.569 | 0.5799     | 2-Stage Baseline              |
| 0.539             | 0.572 | 0.5830     | Added Item2Vec ItemMF         |
| 0.547             | 0.581 | 0.5930     | Added ItemCF UserMF           |
| 0.550             | 0.585 | 0.5970     | Added variation to ItemCF     |
| 0.552             | 0.587 | 0.5985    | Ensemble(LightGBM + CatBoost) |

<br>
# Feature Importance

