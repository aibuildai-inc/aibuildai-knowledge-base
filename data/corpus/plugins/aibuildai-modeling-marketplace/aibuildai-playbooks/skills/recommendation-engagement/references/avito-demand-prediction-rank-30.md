# Our 30th Solution: In which our heroes tried Quantum Gravity, Adaptive Noise and other cool stuff…

Competition: avito-demand-prediction
Rank: #30
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/60006

This was a great competition, with the most variety of data I have seen in kaggle, maybe a time series competition with the same kind of data present here would be great next 😄 Now to our solution

**Features**

*Image Features* : Extracted simple features(all from public kernels), Imagenet pretrained VGG16 to SVD, NIMA(Neural Image assessment) avg and std from Mobilenet and NASnet, SVD on https://www.kaggle.com/the1owl/natural-growth-patterns-fractals-of-nature.

*Text features*: wordbatch ngrams, text stats from public kernels, english text stats, tfidf translated english and russian , Several different embeddings(Fasttext self-trained on description, wordtovec self-trained on description, 3 embeddings from https://github.com/deepmipt/DeepPavlov/blob/master/pretrained-vectors.md preprocessed text according to how the embeddings was trained, english GLOVE wiki vectors, russian fasttext wiki vectors), SVD and LDA on description and title, Naive Bayes Tfidf, Sentiment 

*Tabular features*: All features on public kernels, normalized log price by groupby param_2 train+test+test &amp; train_active with the idea that relative price difference between items in the same category matter, binning of 'norm_price' by different total number of discrete values, count of 'norm_price' bins by param_2 with test &amp; train_active, test &amp; train_active average item count per day of 10+ categorical features by using test &amp; train period using from &gt; to time, Target Encoding of all categoricals with smoothing and noise, almost 60 different aggregate counts of items groupby a set of categoricals and unique counts of categoricals grouped by different sets of categoricals including test and train active, impute by prediction with NN image top 1

**Models**

Our final submision was based on stacking and weighted averaging, it provides the boost we needed from our seemingly poor performing models, even the best ones.

19 Level one models that consist of pure RNN , LGBM, Ridge, Lasso, Elastic Net, Random Forest Regressor, XGB , FM-FTRL, RNN + categoricals, RNN + categoricals + continuous, RNN + categoricals + continuous + images.

10 Level two models build on top isotonic regression k-fold transformed level one predictions, consist of Linear Regression, LGBM, XGB, RNN + categoricals + continuous.

3rd Level model is just a Linear Regression with isotonic regression

Final submission a weighted average with best RNN + categoricals + continuous + images model and 3rd Model.

Best models is LGBM- two of them with very uncorrelated predictions, both at 0.2203 Public LB trained with entirely different subset of features listed above, NNs with all type of feature 0.2207 Public LB

**Notes**

We had two distinct RNNs. One implemented by Chin which is just the best architecture adopted from toxic comment classification and the other implemented by @Pavel and @Andres. The later can be found here: https://github.com/antorsae/avito-demand-prediction and didn't use as many features as described above, and it has a few different features.

**Things that didn't work**

Discretizing predictions aka "Quantum Gravity": As discovered by one competitor, predictions are very discrete, so we built a list of predictions grouped by category and added a post-processing layer to convert them to discrete values based on proximity and strength of discrete probability. We dubbed this approach _quantum gravity_ but although the name was cool it didn't work.

Noise: We added this the last day of the competition so our findings were inconclusive. We implemented "swap noise" and "smart noise", the first just swaps a fraction of columns by values of the same column in different samples, whereas the second picks samples to swap columns from whose prediction is similar to the current sample.
Adaptive noise: We saw that controlling the rate of noise was very delicate and if set too low (e.g. `-fnr 0.1`)  the network would eventually overfit, and setting it too high (e.g. `-fnr 0.3`) would make the network converge very slowly or not converge at all; so we added a callback to adjust noise rate based in a target rmse. We didn't have time to test it properly.

Image pixels: We implemented computing image features and optimizing them in multiple networks (all controlled by command line), and with support for freezing layers; while our initial tests showed promise, we did not have time/GPUs to run it at the end.
