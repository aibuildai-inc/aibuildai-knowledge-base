# 3rd Place Solution (Triplet loss, Boosting, Clustering)

Competition: shopee-product-matching
Rank: #3
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238515

Hi everyone, here is 3rd place solution. First of all, thanks to Kaggle and Shopee for great competition. There was a lot of ways to improve and I believe, no one discover all of them, so there was no "perfect" solution. My solution consists of 3 main parts

## General approach and fast baseline

Let's build different (not trainable) products representations: efficient net embeddings, BERTs, TfIdf .. everything, you think should perform well and search for nearest neighbours on fixed radius, threshold is selected by CV, commonly it's optimal F1 score threshold + some small margin. After that for each object we union candidate neighbours from all embeddings and build a sample of pair objects with a binary target - if pair is a real duplicate or not. In my case train sample was about 3M points pairs with 4% duplicate rate. On that sample I build simple GradientBoosting model (catboost in my case) using next features, calculates separately for each embedding:

- pairwise distances (cosine, euclidean and etc)
- density around both points (frequency of points on different radiuses)
- points ranks

Finally I have about 500 features. After model is tranined, just threshold candidate points by probability of duplicate, that is searched by CV (for test I just search by public LB). This scheme gives around 0.76+ LB score, and takes about 30 min to inference using embeddings: Efficient Net B2, Indonesian BERT cahya/bert-base-indonesian-522M, Multi language BERT setu4993/LaBSE, multiple TfIdfs with different tokenizers and CLIP

## Finetune embeddings using metric learning

I also added finetuned Efficient Net, ViT (DINO) and 2 BERT (cahya/bert-base-indonesian-522M, setu4993/LaBSE) models using the same approach:

- triplet loss (margin=0.1)
- each epoch I create duplicate pairs from label_groups, in each batch one pair from label_group is inserted. Batch size 128 was used, so I get 64 duplicates per batch and I compare each of them with 5 random points
- 5 folds CV, when I calculate validation score, I was looking for candidates not only in validation fold, but in full train sample. This scheme reduces CV/LB gap and make it more correlated. My CV models have CV score around 0.71, BERTs - 0.67. Also this approach is good to construct out-of-fold prediction, which is important to fit 2nd level models (GBM described above in my case). For test set I perform inference of all 5 models from CV. Finally, I thought, that better way is to fit one model on full test with no validation for faster inference, but have no time for all of models (made it just for longest inference DINO model) 

I have also tried ArcFace approach, but models scored much less on CV, and it was surprise for me, because ArcFace was common way to finetune in this competition. Maybe I did something wrong here .. Adding finetuned models to general scheme gives me 0.781 LB score

## Postprocessing (clustering)

I perform clustering, not over embeddings but over pairwise distances. Distance (better say - similarity) for me was GBM probability of duplicate estimation. If there was no forecast for product pair, assume probability of duplicate eq 0. I took the idea of agglomerative clustering and slightly modify it fot current task. We start from each single point as a cluster and merge them until average cluster size become equal threshold (for train sample avg cluster size eq 2.8, for LB best size was 2.6). If after clustering point stay single - merge it to nearest cluster. This algorithm boost me to 0.79 after a lot of parameter search

Few words about optimization: all my models was trained and inferred in half precision (torch AMP). For image model I used NVIDIA DALI for reading and resizing images, for GBM models inference I used Rapids ForestInferenceLibrary. Without everything it becomes impossible to inference all those models in 2 hours.
