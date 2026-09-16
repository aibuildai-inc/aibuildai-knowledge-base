# congrats to the winners

Competition: home-depot-product-search-relevance
Rank: #1
Source: https://www.kaggle.com/c/home-depot-product-search-relevance/discussion/20427#116960

We also would like to share the basic concepts of our solution and share our experiences. First of all thanks to Homedepot for hosting this wonderful competition! 

As many have noticed, this dataset rewards mixing of all kinds of features and modeling approaches which is most easily achieved in teams. No matter what exotic feature engineering and modeling we tried, there was hardly any case that negatively affected the CV score, so we kept on adding models and features until we ran out of time. Our CV strategy worked very well and gave us the confidence to create large L1 and L2 ensembles and adding a third layer of linear stacking. 

We were quite surprised to end up on place 2 on private as we were only 5th place on public! We did not use productuid or id by itself as a feature, so maybe that helped. We noticed the trend and jump in average relevance for different id ranges, but we only accounted for it using a binary variable id>160,000. 

**Data cleaning/Preprocessing**

- Spell correction using posted Google corrections and using Norvig correction on corpus built from titles, descriptions and reddit comments.
- Spell correction of queries using manually created dictionaries published in the forum
- Removing all words in brackets from title (battery not included)
- Splitting of title and query by prepositions (in, for, with, etc.) and inversing order, to get important words to the end
- Normalizations of units

**Feature Groups**

- Word match counts and positions on Wordnet synonym, hyponym and hypernym expanded titles and queries

- Attribute extraction and matching for product titles/attributes and queries
 - Brand
 - Color
 - Material
 - Sizes, volumes
 - Power, voltage, current, btu

- Jaccard and Dice distances between query, title, description
- Char n-gram TF-IDF, SVD, NMF based cosine distance and total common term counts
- Extraction of 1, 2 and 3 important nouns in queries, product titles (using noun position) and all nouns in product descriptions (using nltk pos tagger) and quantification of important noun similarities (L1, L2, cosine distances). Incorporated important and unimportant words dictionary published in the forum
- Longest common subsequence between title, brand and query

- Average query and product title, description, attribute, important noun, Wordnet expanded syn/hypo/hypernym similarities using pretrained word2vec models and GloVe word embeddings
- Glove word embedding centroids and distances based on various corpora (pretrained, generated from query+title+description, enriched with Reddit DIY and Home Improvement comments, enriched with Wordnet synonyms)
 - Relaxed word movers distance
 - Relaxed word movers distance (weighted by query/title word position)
 - Cosine distance
 - Euclidean distance
 - Centroid vectors of query, title, descriptions as features
- BOW, TF-IDF, LSA based features
 - Word N-gram TF-IDF cosine distances (N=1,2,3)
 - SVD cosine distances (50 dim, 100 dim, 300 dim)
 - NMF cosine distances
 - Raw SVD vectors as features
- Aggregates by query and product
 - Standard deviations, rank transformations, percentiles of distance features
 - L1,L2, cosine distances to query and product centroids of SVD/NMF/Glove
- Statistics features
 - Num occurrences of query in whole dataset
 - Num occurrences of product in whole dataset
- Exploiting the id anomaly
 - Boolean indicator id>160,000
- Feature interactions
 - 2-way polynomial interactions between top ranking features

**Cross-Validation**

We used 3 runs of 3-fold cross validation: 2 runs with disjunctive queries (query does not occur in validation sets), 1 run with disjunctive product id, to roughly match the proportions of unseen queries and product ids between train and test.

**Modeling and Ensembling**

- About 200 L1 models based on various feature subsets, trained with XGB, RFR, ETR, Neural Nets, Random Trees Embedding+Lasso/Ridge, ETC (Regression via Classification)
- L2 Stacking with Bayesian Ridge, 10X bagged Neural Net and 10X bagged ETR, trained for each L1 CV run
- L3 Stacking with ridge regression
