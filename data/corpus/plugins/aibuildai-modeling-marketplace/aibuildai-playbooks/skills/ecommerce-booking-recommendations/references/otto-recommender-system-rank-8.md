# 9th Place Solution

Competition: otto-recommender-system
Rank: #8
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/383130

I'd like to thank the Kaggle staff and the OTTO team for organizing this interesting competition!

I'm relieved that I was able to make it through to the end, even though my score did not improve much in the final stages of the competition.

Here is my solution

## Candidate generation
- **re-visit** - all items from the session's history
- **co-visitation matrix** 
   - any2any, click2click, click2cart, click2order, (click or cart)2order, (cart or order)2order, e.t.c.
   - I tried various patterns to create the co-visitation matrix. 
e.g. actions that do not take time into account, actions immediately after an action, within 5 or 10 actions, within 5 or 10 minutes, e.t.c.
- **create click2click(only consider the next item) graph and apply ProNE** 
   - The idea came from the hypothesis that items clicked on immediately after an item is clicked are similar to each other.
   - I created a two-column DataFrame (an item and the item clicked immediately after) and ran it through ProNE. The number of dimensions of ProNE was 1000, and the more dimensions I increased, the more accurate the candidate recall became.
   - Retrieve top-k aids by ProNE embeddings (Used cuml.neighbors.NearestNeighbors and metric='cosine')
- **word2vec**
   - Trained word2vec model with aid sequences (Used gensim, size=50 or 100)
   - Retrieve top-k aids by w2v embeddings (Used cuml.neighbors.NearestNeighbors and metric='cosine')

## Re-Ranking
### Model
- LGBMRanker (lambdarank) 
- I created one model each to predict clicks, carts, and orders.

### CV
- I created validation sets with the host's old version scripts
- I created 100 candidates per session when training
- candidate recall
   - click: 0.6622
   - cart: 0.5113
   - order: 0.7059
- CV 
   - click: 0.5601
   - cart: 0.4414
   - order: 0.6664
- I created 300 candidates per session when inferencing
- Public LB: 0.603, Private LB: 0.603

### Feature
- **session features**
   - type count by session (type='clicks' or 'carts' or 'orders') 
   - number of unique types by session (type='clicks' or 'carts' or 'orders') 
   - type mean by session ('clicks'=1, 'carts'=2, 'orders'=3 and mean by session) 
- **aid features** 
   - type count within all sessions (type='clicks' or 'carts' or 'orders') 
   - type count within test sessions (type='clicks' or 'carts' or 'orders') 
   - click to cart rate within same sessions, cart to order rate within same sessions, click to order rate within same sessions
- **session x aid features**
   - co-visitation count, rate, time-weighted count, rate, e.t.c
   - **similarity**
      - cosine similarity between candidate item and last item of the session (w2v, ProNE)
      - cosine similarity between candidate item and the second item from the back of the session (w2v, ProNE)
      - cosine similarity between candidate item and all items of the session (w2v, ProNE)
      - click2click Jaccard index score

## What did not work well
- Create candidates with node2vec
- GRU
- RecVAE
- SAR
- BPR
- Pseudo Labeling
