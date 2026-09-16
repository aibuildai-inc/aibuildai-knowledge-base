# 12th Place Short Summary

Competition: foursquare-location-matching
Rank: #12
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/336051

Thank you to Kaggle and Foursquare for hosting this competition. It was a fun and interesting challenge. Our team was not aware of the leak, but our solution is most likely affected by it. Our final approach is a weighted average of XGBoost, CatBoost, and a language model-based NN. This is a quick summary of some of the things that our team did differently from the solutions shared by other competitors.
 
## Candidate Generation
[Kosuke’s](https://www.kaggle.com/sakami) candidate generation is one of the main components of our solution. In the end, we achieved a max IoU of 0.97675 using 25 candidates per id. 

* We extracted candidates in three different ways:
    1. latlon top-k
    2. name & latlon embedding top-k
    3. category & latlon embedding top-k
* name & latlon and category & latlon are simple concatenations of their embeddings.
    1. We used the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) to create embeddings.
    2. There was a lot of data with the same name all over the world, e.g. Starbucks. The embedding is concatenated to extract places with the same name that are close in distance. This took advantage of the fact that the order of magnitude of the latlon L2 distance is much smaller than the order of magnitude of the name embedding L2 distance.
    1. The categories embedding was the average embedding for each category.
* We picked 15 latlon neighbors, 15 name & latlon neighbors, and 10 latlon & category neighbors. Then we picked the top-k samples with the smallest distance among them. Note that we extract more than the pre-defined minimum number of candidates for each id.

## Postprocessing
Postprocessing gave us a boost on the LB. We created a soft adjacency matrix from the graph where nodes are places and edges are match probabilities. First, we found all 1-hop paths that exist using edges with a predicted probability greater than 0.5 (this is equivalent to the postprocessing used in most public notebooks). Then, we found all 2-hop paths that exist using edges with a predicted probability greater than 0.9. We repeated this for 3-, 4-, and 5-hop paths and thresholds of 0.95, 0.998, and 0.999, respectively. In the end, we predicted that two places are a match if there are at least one of these n-hop paths between the places.
