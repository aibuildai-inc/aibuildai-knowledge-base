# 3rd solution - supplement to the post-processing section

Competition: happy-whale-and-dolphin
Rank: #3
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/320191

First of all congratulations to all the winners 🎉， and thanks to kaggle for hosting such an interesting competition. 

A lot of post-processing strategies were tried in this competition，but most are disappointing.
* re-rank, accelerate with gpu and segmentation strategy. （not work）
 * https://arxiv.org/pdf/1701.08398.pdf 
*  dba, or neighbor blending. (not work)
 * https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
* gnn-re-rank, based on graph convolution. (not work)
 * https://arxiv.org/abs/2012.07620
* some feature dimension processing strategies：PCA(part/whole), SVD(part/SVD), RMAC
*  bayesian search weights
 * before 0.875, it can be greatly improved by searching
 * but through repeated iterations, when the model reaches 0.880, the effect of search and concat is equivalent

Reference Code：
* https://github.com/PyRetri/PyRetri
* https://github.com/Xuanmeng-Zhang/gnn-re-ranking
* https://github.com/JDAI-CV/fast-reid
