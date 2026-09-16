# 2nd Place: with help from NNs.

Competition: playground-series-s3e26
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s3e26/discussion/464887

With this competition I set out to improve my knowledge of neural networks. I had trouble getting good results with them in past competitions, often finding they performed worse in the private leaderboard, even when added to ensembles of GBDTs. However, this time I think they were the secret sauce that helped me get 2nd place.

My solution is pretty straightforward. Just a combination of XGBoost, LightGBM, and neural network predictions, with another neural network used to stack them together. The XGBoost and LightGBM predictions were nothing special, just an average of predictions from 10 different sets of hyperparameters found by Optuna.

For the initial neural network predictions, after reading through a bunch of papers and iterating on the techniques described within, I found that the piecewise linear encoding (PLE) technique described in [this paper](https://arxiv.org/pdf/2203.05556.pdf) greatly improved my score. 

I applied PLE to each continuous feature, an embedding layer for the edema and stage features, and fed in the remaining binary features as is (after converting to 0/1). The structure looks sort of like this:
[asdf]
The actual network included some additional layers after each PLE input, before they were concatenated together, but to keep the diagram simple I did not include them. With this neural network alone I was able to get a score around .401 on the private LB, good enough for top 10%. Despite its solid individual performance, its addition to the stack only added around a .001 boost to the score, but I'm just glad it improved the stack at all.

For the stacking neural network, I started with the idea of getting the network to learn a weighted average of input predictions, which looked something like this:


Through some more experimentation, I found that the network achieved better scores by
1. Giving each class probability within a prediction its own weight, rather than computing a single weight and applying it to each class probability, and
2. Calculating each prediction's weights using all 3 predictions, rather than only itself.
With these changes the network looked more like this:

Again, for the sake of simplicity, there are some additional layers not included here. This stacking architecture gave a boost of .004 compared to taking a simple average.

Overall, I had a lot of fun working on this competition. The 2nd place was a nice bonus, but I'm more grateful from the knowledge acquired in the process of researching and trying out neural networks. For anyone else interested in recent tabular neural network innovations, I would recommend taking a look at https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html. There are plenty of ideas using neural networks that I either wasn't able to try or wasn't able to get working for this competition, but definitely look forward to trying these ideas and more in future competitions.

Happy new year!
