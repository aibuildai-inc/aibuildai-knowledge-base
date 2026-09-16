# 2nd place solution

Competition: tabular-playground-series-jun-2021
Rank: #2
Source: https://www.kaggle.com/c/tabular-playground-series-jun-2021/discussion/250060

First, I would like to thank the notebooks shared in this competition, I wouldn't be here without them!
My solution is simple ensemble (weighted averaging) from some models:
1.  NN + GBT similar from [NNs+GBTs](https://www.kaggle.com/hiro5299834/tps06-nns-gbts-optimization)
2. Simple NN (Embeddings -> Conv1D -> Residual) + KNN features thanks to kernel [keras tuner + knn feature](https://www.kaggle.com/remekkinas/keras-tuner-knn-features-simplex-optimization)
3. Simple NN (1DCNN +2DCNN + Residual)
4. LightAutoML
