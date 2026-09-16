# 5th place (yet) solution (Carno & 2U & Jiahong's part)

Competition: otto-recommender-system
Rank: #4
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/382783

Same as other competitors who benefit from @radek1 and @cdeotte, we thank their great devotion to this competition. We'd also like to thank the organizers and Kaggle, and I hope what happened in this competition will end up with a satisfactory endding.

## TL; DR

Our solution is a combination version from two teams' previous solutions. Before team-merge, our train-valid splits, recall methods, feature sets and rerank models are all different. We not only simply ensemble our independent submission files, but also exchange features for further improvement.

## Ensemble

Here, we use lower case letters to denote a submission version **(a, b, ...)**, which can be the 5-fold ensemble of one reranker, or an ensemble of two or more submissions. **S1** and **S2** are public validation data split and private regenerated validation data split. **R1** and **R2** are two different recall methods. **F1** and **F2** are two different feature sets, and **F'1** and **F'2** are the important feature sub-sets. **XGB** and **CBT** denotes xgboost binary classifier and catboost ranker. **"+"** denotes score ensemble, which means we average the raw output from multiple model to rerank candidates. **"&"** denotes index ensemble, which means we assign index-score for the first 20 candidates from 1.00 to 0.05, with 0.05 step, and then we use the summation of index-scores to rerank candidates. **"*"** denotes weight during ensemble.

Our final solution ***i*** whould be:


***c*** = a * 0.45 & b * 0.575 
***d*** = d1 * 0.5 + d2 * 0.5 
***e*** = e1 * 0.5 + e2 * 0.5 
***g*** = c * 0.5 & d * 0.4 & e * 0.6 
***h*** = (c * 0.5 & d * 0.5) & f * 0.5 
***i*** = g * 0.6 & h * 0.5



|        | data splits | recall methods | feature set | model    | importance feature set |
|--------|-------------|----------------|-------------|----------|------------------------|
| a      | S1          | R1             | F1          | CBT      | F1', F1''              |
| b      | S2          | R2             | F2          | XGB      | F2'                    |
| d1, d2 | S2          | R1             | F1+F2'      | XBG, CBT |                        |
| e1, e2 | S2          | R2             | F2+F1'      | XBG, CBT |                        |
| f      | S2          | R2             | F2+F1''     | CBT      |                        |

## Recall methods

### R1

This recall methods is developed based on public co-visitation matrix notebook (4 matrix: clicks, carts, orders and buy2buy), and optimized by numba. The detailed numbers will be released with code.

### R2

PLACEHOLDER

## Feature set

### F1

F1 includes statistical features and model trained features. Same as most teams, we use sum, max, min and mean of interaction history and co-visitation score to summarize the sessions, the items and the interactions. The importances of most statistical features are not significant.

In trained features, we used BPR, ALS and LMF from `implicit` package, W2V from `gensim` package and [SAS](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382783#2124459). In all algorithms mentioned here, we can get the embedding of items, so we use the inner product of candidate embedding and session latest average embedding as interaction features. In BPR, ALS, and LMF, we can get the embedding of both sessions and items, so we additionally use the inner product of session embedding and candidate embedding as interaction features. 

### F2

PLACEHOLDER

### Importance

We use feature importance from reranker model to decide which features to exchange.
