# Private 11th (Public 9th) Place Solution Summary

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #11
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551424

Firstly, thanks to Kaggle organizers and my team mates.
I will briefly share our solution.

## Overview
Our pipeline mainly follows flagembedding https://github.com/FlagOpen/FlagEmbedding. It can be divided into 2 stage: retrieval and reranker. 

@sayoulala : https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519 Thanks for sharing, that's where a lot of our design of pipeline comes from.

## Retrieval
In addition to training data, we also use external datas from the disscussion https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/533764 and https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/541222, thanks for sharing. For convenience, we refer to them as ext1 and ext2.


We divide the training process into 3 stage. First, we train the model using ext1, then we train the model using ext2, finally we train using training data. 


We observe that, maybe the ext1 has noise, so when we train the model using ext1+training data, the performance is worse than 2-stage training. And for ext2, because it lacks the construct name and subject name, we continue using the multi-stage training strategy. 


## Reranker
We obtain three trained retrieval models from each of the above three stages. And then We use these three models to create hard negative data for ext1, ext2, and training data, respectively. Then we also divide the training process of reranker into 3 stages, in each stage we use corresponding hn data to train.

## Some Details
When we train the 3rd stage of retrieval, we use negative cross device to enlarge the number of negatives for each positive. When we use it, public score 0.567->0.574.

We use Qwen2.5-32B for retrieval and reranker. Because of the limit, we only rerank top 18 in inference. 

| Model | Public LB |
|----------|----------|
| retrieval   | 0.582   |
| retrieval+reranker   | 0.623   |

## Code
The main training code is provided in https://github.com/MilchstraB/Eedi
