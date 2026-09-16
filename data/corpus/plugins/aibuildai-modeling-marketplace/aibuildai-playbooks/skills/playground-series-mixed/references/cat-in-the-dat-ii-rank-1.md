# 1st Place Solution

Competition: cat-in-the-dat-ii
Rank: #1
Source: https://www.kaggle.com/c/cat-in-the-dat-ii/discussion/140465

Congrats to all the participants in this great and challenging tabular competition! Thank you  Kaggle for organizing this competition.

My solution is very simple, NN plays a key role. The categorical features, especially with high cardinality are very suitable for neural network to exert its power.

### About Models
4x NN,  1x Catboost, and blend of some of public kernels.  All NN models on same features,  and one of them resulted in 0.78672 on the public LB. Thanks to @sergey and @siavash for sharing the public kernels, I used them in blending.

### Feature engineering for NN models

1. All features are converted into category type then Label Encoding
2. Ordinal Encoding: ord_1~ord_5 -&gt; ord_1_en~ord_5_en

That’s all. I have also tried using various other methods to process the features but they all lead to overfitting.

### NN Models
NN uses several state-of-the-art models for CTR prediction, including CIN in xDeepFM, PNN, Cross in DCN, AutoInt, etc.

1.Linear+DNN+CIN (0.78672 public LB)
2.FM+Cross+PNN (0.78655 public LB)
3.FM+DCN+DNN (0.78652 public LB)
4.Linear+DNN+AutoInt (0.78665 public LB)

There are many components available for feature extraction on tabular data and they can be combined with various ways. It is a huge workload to trail by coding from scratch every time. Deeptables greatly simplifies this job with only a few lines of code.

  [https://github.com/DataCanvasIO/deeptables](https://github.com/DataCanvasIO/deeptables).

[I have shared the NN code here.](https://github.com/DataCanvasIO/DeepTables/blob/master/deeptables/examples/Kaggle%20-%20Categorical%20Feature%20Encoding%20Challenge%20II.ipynb)
