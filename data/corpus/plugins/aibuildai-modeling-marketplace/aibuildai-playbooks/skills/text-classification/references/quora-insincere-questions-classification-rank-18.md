# 18th place solution from 300-th at Public LB

Competition: quora-insincere-questions-classification
Rank: #18
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80696

Thanks Quora for great competition and thanks all who participated and contributed here.
We were surprised at this result since we were in bronze in Public LB.
Our (me and [@hattan0523][1] ) kernel is here.
https://www.kaggle.com/kentaronakanishi/18th-place-solution
The local CV of our model is around 0.696-0.698.

Our points are below:

- simple 2 layers RNN model with units=96 by 5 fold
- use semantic bernoulli dropout for embedding layer
- remove any filter when using keras tokenizer
- batch size control to get larger batch size in later epochs
- cut data length at max length in a batch
- learn embedding weights only at last epoch
- lots of lucks by using my wedding anniversary as seed

We don't use in our case:

- capsules: contribute but take too much time
- attention: only effect val_loss and no change in CV score
- CNN based model: It worked but increasing rnn width is better for us
- GBDT models (XGboost, LightBGM): They don’t achieve our desired CV score, and a lot of time are needed to run. Thus, we don’t use an ensemble or stacking method. 
- word typos: a kernel (https://www.kaggle.com/sunnymarkliu/more-text-cleaning-to-increase-word-coverage) shows many typos of the data set at In[18] of kernel. We rechecked and modified them. This is good for public LB, however not good for private LB.

All questions and suggestions are welcome.


  [1]: https://www.kaggle.com/hattan0523
