# 10th place solution - Meta embedding, EMA, Ensemble

Competition: quora-insincere-questions-classification
Rank: #10
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80718

Here are all my submissions. 1: keras, 2-4:pytorch

1. [Projection meta embedding and EMA(version 1/4)](https://www.kaggle.com/tks0123456789/projection-meta-embedding-and-ema?scriptVersionId=7916644), 0.69480(Public)

2. [PME_EMA 6 x 8 pochs(version 2/14)](https://www.kaggle.com/tks0123456789/pme-ema-6-x-8-pochs?scriptVersionId=10163202), 0.69568
3. [PME_EMA 6 x 8 pochs(version 10/14)](https://www.kaggle.com/tks0123456789/pme-ema-6-x-8-pochs?scriptVersionId=10224276), 0.70551, 0.70964(Private)

4. [PME_EMA 6 x 8 pochs(version 14/14)](https://www.kaggle.com/tks0123456789/pme-ema-6-x-8-pochs?scriptVersionId=10275816), 0.70061, 0.70921

## Preprocessing
Separating punctuations only. Spell correction didn't work for me.

## Model structure
**Average** ensemble of 6 models of the same network.

    Embedding(max_features, 600)
    Linear(in_features=600, out_features=128, bias=True)
    ReLU()
    GRU(128, 128, batch_first=True, bidirectional=True)
    GlobalMaxPooling1D()
    Linear(in_features=256, out_features=256, bias=True)
    ReLU()
    Linear(in_features=256, out_features=1, bias=True)

## Projection Meta Embedding(PME) 

Meta embedding is a method for combining multiple pretrained embeddings and dicussed in [3 Methods to combine embeddings](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/71778).
PME is **Unweighted DME ([4])+ ReLU**. It concats several frozen pretrained embeddings and project to a lower dimensional space by linear layer with ReLU activation.



## Exponential Moving Averaging of weights(EMA)

It caluculates exponential moving average of weights during training. It is usually done on a minibatch level. I chose 10 updates per epoch for speed.
[1], [2], [3] use EMA.


## Tuning for ensemble

### n_embed: #of pretrained embeddings in a single model.
I tried n_embed=1, 2, 3, 4, and 2 is best. Using 4 embeddings is better for a single model F1, but lacks model diversity, which cause worse ensemble performance.

### epoch
The followings are mean F1 of 10-fold CV. The best epoch is 5 for a single model and 8 for ensemble.
[Image]


[1] [A. Tarvainen and H. Valpola(2017) Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.](https://arxiv.org/abs/1703.01780)

[2] [Yu, A. W., Dohan, D., Luong, M.-T., Zhao, R., Chen, K., Norouzi, M., and Le, Q. V.(2018) QANet: Combining local convolution with global self-attention for reading comprehension](https://arxiv.org/abs/1804.09541)

[3] [Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi (2016) Bidirectional attention flow for machine comprehension.](https://arxiv.org/abs/1611.01603)

[4] [Douwe Kiela, Changhan Wang, Kyunghyun Cho (2018) Dynamic Meta-Embeddings for Improved Sentence Representations](https://arxiv.org/abs/1804.07983)
