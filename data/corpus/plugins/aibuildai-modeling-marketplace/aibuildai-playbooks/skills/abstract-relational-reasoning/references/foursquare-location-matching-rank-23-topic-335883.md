# 23rd place solution (arcmargin)

Competition: foursquare-location-matching
Rank: #23
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/335883

Congratulations to all medal winners and thanks to Kaggle and Foursquare for such an interesting competition. 

My solution is based on arcmargin models to extract candidate pairs and LigthGBM for matching. 

**Blocking**: Generate pair candidates with arcmargin model. This model is trained with arcmargin loss ([https://arxiv.org/pdf/1801.07698.pdf](url))  

Features: 

- TFIDF (char level of (1, 3) ngrams) of name, address, zip code. 

- TFIDF (word level) of categories. 

- Onehot of city, country, state and the domain of the url. 

- Binary features: has_phone, has_url, etc. 

- Latitude and longitude encoded with a trick explained below. 

Model: 

All the inputs are concatenated and followed by two blocks of Dense-PReLU-BN-Dropout. The output of the last block is the input to the arcmargin layer. 

- Size of embedding is 512.  

- It’s important to use L2 regularization in the arcmargin layer (it has a lot of parameters, >200M when is trained with >500k points of interest). I use L2=1e-4 

- Margin warmup from 0.2 to 0.8. I think it’s not very important, but the model converges faster. 

Results: 

These models generate a set of pairs with a max IOU of >0.98.  

Public LB: 0.89. 

**Matching: LightGBM models** 

Features: 

- Cosine distance between embeddings.  

- Euclidean, Manhattan, Harvesine distances between lat, lon. 

- Levenhstein, Jaro-Winkler, LCS, distances/similarities ([https://github.com/seatgeek/thefuzz](url)). 

Results: 

Public LB: 0.908 

**Final submission**: 

- New arcmargin model based only on text features with XLM-Roberta. Public LB: 0.917 

- New features in LGB models: Embedding rankings, cosine distance to the nearest neighbor, etc. Public LB: 0.923. 


**Latitude and longitude encoding trick**: Because the latitude and longitude features are of high precision, it is necessary to encode them in some way to make them useful for the arcmargin model. I have used a technique like positional encoding of transformers to embed these high precision features in vectors. The code used is: 

````
emb_size = 20 
precision = 1e6 
 
latlon = np.expand_dims(df[["latitude", "longitude"]].values, axis=-1) 
 
m = np.exp(np.log(precision) / emb_size) 
angle_freq = m ** np.arange(emb_size) 
angle_freq = angle_freq.reshape(1, 1, emb_size) 
 
latlon = latlon * angle_freq 
latlon[..., 0::2] = np.cos(latlon[..., 0::2]) 
latlon[..., 1::2] = np.sin(latlon[..., 1::2]) 
latlon = latlon.reshape(-1, 2 * emb_size) 
````
