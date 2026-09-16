# 11th place code and solution

Competition: foursquare-location-matching
Rank: #11
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/335924

First of all I would like to thank the team member and, participants and Foursquare.
Except the data leak, the problem design was very interesting and a good competition with many possible approaches.

Our team may have been too pure to suspect the data leak, (because the host explicitly declared that there was no overlaps).

ATTENTION: This notes was written before the data leak was known, and some points may seem silly for now, but we will share it.

### Links
- Training(XGB) notebook: https://www.kaggle.com/code/iiyamaiiyama/fs-training-4-ensemble-0705/notebook?scriptVersionId=100082426
- Inference notebook: https://www.kaggle.com/code/iiyamaiiyama/fs-inference-no54-spatial-no-country-07045/notebook?scriptVersionId=100093618 
- NN models traing code (github): https://github.com/heartkilla/mcd-kaggle-fsq

### Summary
- Metric learning multi-input(concatenated text -> BERT + lat/lon) NN model.
- Generate candidate by NN model embeddings and spatial nearest neighbours.
- Create 2nd stage binary classification XGB models for each candidate rank.

### Validation strategy
- In the beginning we split training data in 4 folds, and trained BERT(train:valid = 3:1). Then we pick up one fold(validation), and split it further in 4 folds for 2nd stage XGB model training.
- After a while, we found that we can get much better results if we trained BERT using all the data. In this case we train XGB with leaked data and it may perform badly for the test data. Maybe the quality of embeddings is more important than traning a leak-free XGB model.

### Preprocessing
- Fill `NaN` texts with nearest(in longitude and latitude) 5 points for BERT models.

### 1st stage: Create candidates
- BERT embeding candidates
    - Architechture(BERT + lat/lon)
        - Concatenate BERT model output with normalized lat/lon values and connected to the FC layer(320).
        - FC layer(320) embeddings were used for similarity.
  	- Loss: ArcMarginProduct
  	- Training
 		- Gradually increase Margin during training(0.2 to 0.8)
 	    - 40 epoch training(24-48 hours to train)
    - Emsemble
        - Concatenate multiple BERT models embeddings
 			- `xlm-roBERTa-large`, `sentence-transformers/LaBSE`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, `remBERT`
    - DBA/QE
        - Weighted DBA/QE for better embedding.
    - Create candidates
        - For each id, 50 candidates were created using cosine similarity (faiss)
- Spatial candidates
    - Added candidates from lat/lon using sklearn.neighbors.BallTree.
    - Hoping to add candidates that were missed by BERT embeddings.

### 2nd stage: GBDT models for each candidate rank
- Many public kernels train GDBT model after taking N(e.g.20) neighbors from each point by Bert embeddings and creating a dataset with size len(test_df)*N. We create XGB models for each rank of the candidates. We first search the rank point(excluding itself) from each point and create a dataset with a size of len(test_df) to train one XGB model. Similarly, we create another XGB model from the second rank point, and so on. Finally we created 50 models. This is very efficient approach in memory, training time and accuracy.
- Sample weights were set according to the number of POI matches.
- ForestInference was used during inference. It was very fast and helpful.

### Post process(dijkstra)
- To generate the set of matched point, we defined a graph from the output of XGB models and then used Dijkstra's algorithm. We add edges x -> y for each point x and its candidate y and set the weight to 1.0 - (prediction of XGB model). We then generate the set of matched point by collecting all points whose distances from the focussing point are smaller than a threshold. This process reduces the number of FN.
- Before executing Dijkstra's algorithm, we make the graph undirected by averaging the weigths of  forward edge x -> y and inverse edge y -> x.
- We tried various binary operations (sum, prod, max, etc...) of weights to change the distance on the graph. The usual summation works best for us. 

### LB vs CV
- As we discussed before, our approach is pretty leaked. Our best CV is 0.997. But we can see almost linear CV-LB relationship with this leaked pipeline. This approach caused problem, our score stopped increasing when the CV score was close to 1.0
- LB vs CV plot

### Hardware
- Our team did not have sufficient GPU, and Google Colab(pro) was not powerful enough for Bert training. So we employed Vast.ai, which is relatively inexpensive compared to GCE or AWS, and allowed us to use good GPU.
- For XGB training, Kaggle Kernel was enough. We can finish XGB training within one hour.

### Score Timeline
- 0.922 baseline: bert-base-multilingual-cased
- 0.928 +ensemble xlm-roberta-base(2 ensemble total)
- 0.931 +add XGB feature
- 0.940 +fill na
- 0.945 +add bert models(4 ensemble total)
- 0.947 +dijkstra postprocessing

###  Not worked for us
- shuffle text augmentation
- reverse geocoding text augmentation
- text unidecode -> deberta-large-v3

### Postscript: When should we notice the leak?
In retrospect, we had several opportunities to know the data leaks.
- We noticed that there are a overlap items of `train.csv` in sample `test.csv`.
- BERT training on all data scored much higher than the single fold model.
- XGBoost parameters which are more aggressive, gave better scores.
