# 16th place solution

Competition: mercari-price-suggestion-challenge
Rank: #16
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50322

Hi Everyone,

Ending final validation, I share my solution.<br>
https://www.kaggle.com/toshik/test-60?scriptVersionId=2459048

My solution is consit of three part.

[1] NN (Embedding-Sum-MLP) part

 I use Chainer in this part. The words in name and item-descriptions are replaced to wordvectors. 
After that, aligned wordvectors are aggregated. They are connected to subsequent MLP and output. Initial embedding matrix is set as Fasttext pre-trained vectors.

[2] NN (Embedding + Convolution) part

 I use Tensorflow in this part. Initial embedding matrix is set as Word2Vec pre-trained vectors. Wordvectors are connects to convolution, MLP and output.
 
 This part is forked from this starter kernel. Thanks to ololo.<br>
 https://www.kaggle.com/agrigorev/tensorflow-starter-conv1d-embeddings-0-442-lb

[3] N-Gram + Regressor part

I use LightGBM, Ridge, FastFM in this part. TfidfVectorizer, LabelBinarizer, and CountVectorizer are used for feature extraction. This makes sparse feature matrix. The price predicted by LighGBM are used for Ridge (Stacking!). At last, output from Ridge and FastFM are averaged.

This part is forked from this public kernel. Thanks to dangtao.<br>
 https://www.kaggle.com/dangtao/go-go-go

In addition, I tried some works (clean name of products, make additional features ... and so on).
Final predictions are logarithmic average.

` [ALL] = 0.25 * np.log1p([1]) + 0.3 * np.log1p([2]) + 0.45 * np.log1p([3])`

I utilize multiprocessing and chunk-reading to save kernel resources (time and memory).

Thank you !
