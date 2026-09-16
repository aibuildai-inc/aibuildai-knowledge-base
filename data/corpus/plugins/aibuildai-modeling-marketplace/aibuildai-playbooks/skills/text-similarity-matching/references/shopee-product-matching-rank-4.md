# 4th Place Solution

Competition: shopee-product-matching
Rank: #4
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238295

## Introduction
Thanks to Kaggle and hosts for this very interesting competition with a tricky setup and lots of room for experimenting and tuning. As always, this has been a great collaborative effort and please also give your upvotes to @christofhenkel and @philippsinger. In the following, we want to give a rough overview of our solution.

## Setup and CV
While initially a split by label_group was doing quite well w.r.t. to LB/CV agreement, later on in the competition with better models and especially with ensembling and post-processing, it was getting harder and harder to find a good CV. 
Consequently, we were looking at three different KPIs: 
1. F1 score evaluated on oof, 
2. F1 score evaluated on oof but also using infold embeddings as noise
3. F1 score evaluated on oof but also using infold embeddings as noise and weighting False Positives with a factor of 2

2 and 3 are known to be leaky, as we are mixing in infold embeddings, which may be easier to differentiate from oof embeddings. Also, they will have better cross linking which adds additional leakage for specific post-processing methods. 1 is not leaky, but with a 5Fold validation, we are only evaluating on less than 7k rows (around 10% of test). Rational parameter tuning is not possible that way and some post-processing methods excel on small sets but fail on larger sets (think about e.g. always picking rank2 prediction). Hence, we generally looked at all those validation methods simultaneously and tuned threshold parameters mostly on method 3.

## Models
In our final submissions, we used Bert based Text Encoder and Image Encoders with different backbones (mainly nfnet, effnet). As many other teams, we did see some improvement in single models when tuning the training and the backbone, but in the blend this effect was almost vanishingly small and we were kind of reaching a plateau there. After the backbone, GeM and Avg pooling was used. The neck is comprised of a linear layer that is used for dimension tuning, followed by batch normalization and PReLu activation. Our embeddings usually had a length around 1024 with larger embeddings yielding a slightly better score on CV/LB. We also added Tfidf (char and word) to the mix of embeddings and even though CV didn't change much it seemed to help a few points on LB. As Tfidf generates very large embeddings and we are constraint by RAM and VRAM in the kernel, we used Random Sparse Projection to reduce the dimensions.

[single_models]

We trained our image and text models with an ArcFaceLoss. ArcFace margin needed quite some tuning depending on the model and we also saw batch size to matter during training. Some models seemed to be very sensitive to the learning rate and as others reported, gradient clipping may have also helped to stabilize the training.

## Ensemble
In our ensemble, we concatenated and normalized image embeddings, text embeddings (bert based) and tfdif embeddings. On each of those vectors, we calculated pair wise cosine similarity and received three matrices (cossim image, cossim bert, cossim tfidf). We combined those three matrices by first squareing them and then taking a weighted average. 

[ensemble]

## Post Processing
While the averaged cossim matrix already gave us a very competitive score in the gold region when using a proper threshold, we still applied several post processing steps to squeeze out a few more points. The biggest contributors were:
- Thresholding
- Rank2 matching (if A has B on rank2 and B has A on rank2, add them to each other)
- Rank2 and rank3 difference is large -> add rank2 id
- If there is a group of X members, and we have another row with the X preds on rank 1-X, make one large group.
- At least one other match (except the cosine similarity of rank2 is extremly low)
- Query Expansion
- Rematching unmatched rows
