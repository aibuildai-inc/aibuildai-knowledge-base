# 7th place solution

Competition: avito-demand-prediction
Rank: #7
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/60026

Hi, dear kagglers.
First of all, thanks to Kaggle and Avito for holding such a wonderful competition. And congratulations to the winners and all Kagglers.

Our final solution was large stacking that involved 64 models in total. It’s quite similar to the post that was posted by KazAnova on the [kaggle blog](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/) before. And after we completed building models, We grouped all the level-0 model into 4 group in order to gain some diversity :

+ All models
+ Branden and Takuoko-Angus models
+ Takuoko-Angus and Yusaku models
+ Yusaku and Branden models

We used the following models for our stacking (level-1 &amp; level-2) : {XGB gblinear, LGBM,  Ridge, Extra Trees, simple NN}. Finally we got 20 level-1 models &amp; 5 level-2 model in consequence. than we just use a simple ridge for level-3 stacking (but excluded the meta-features which had negative weight in ridge, than rerun, until there is no negative weight appeared). It was one of our last submission, it scored 0.2148 on public LB (private 0.2186 / CV 0.208749).

In another submission, I just applied the optim() function in R with BFGS solver on level-1 meta-features to find the best weight to blend, with following formula : x1*model1 + x2*model2 + ...... +x20*model20 + x21. It worked almost same well as our level-3 stacking. It scored 0.2148 on public LB(private 0.2186 / CV 0.2087724). Actually two sub are quite same.


# Takuoko and Angus’s solution 

Takuoko and I merged on the 10 days before merger deadline.

### Feature Enginnering 
+ Applying (min, max, mean, var) to numeric features that was already grouped by some categorical features (e.g. groupby(by=“region”)[“price”].mean())

+ nunique feature 2-way interactions

+ count encoding on categorical features

+ [aggregated feature of categorical features](https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm )

+ [image features : mean, whiteness, dullness of RGB](https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality )

+ Applying PCA / TSVD to OHE categorical features

+ [text stats features](https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author)

+ different word n-gram and char n-gram 

+ applying TF-IDF or not

+ Some features from Angus’s FE (see the attachment)


### Features not work   

+ nunique feature 3-way interactions

+ Applying tsvd on tf-idf text feature or VGG16 features

+ Ohe-hot encoding on categorical features

+ Applying GaussianRandomProjection / FastICA / LDA / SparseRandomProjection on OHE  

+ [fuzzywuzzy features](https://github.com/seatgeek/fuzzywuzzy )


### Modeling

-- tree based model : LGBM, XGB --
Basically we used Bayesian for our tuning. And there is something special in our setting, In LGBM, Takuoko set 0.1 to feature_fraction and low colsample_byleve in XGB. We achieve great success by performing bagging with different seed. We can get about 0.2184 on public LB with single(5 seed avg.) LGBM by this approach.
 
We also used objective=poisson. It needs more time to converge and its performance is not so good, but it gave us some extra boost when we doing stacking.

-- NN --
We constructed a lot of different NN like GRU, Conv1D, Conv2D based on this [public kernel](https://www.kaggle.com/shanth84/rnn-detailed-explanation-0-2246).
We also made a MLP since its score is pretty bad but good for stacking. Until end, We still can’t let our NN model beat the 0.2210 on public. even with BN and fine-tuned Dropout. And that’s pretty frustrated for both of us.

-- Ridge, drop0 model --
Ridge and [drop0 model](https://www.kaggle.com/c/allstate-claims-severity/discussion/26416) also were not good solo, but both of them improved our stacking score.

Drop 0 model was inspired by the link above. We just dropped target=0 and train a LGBM.

## Branden and Yusaku Solution

Branden and Yusaku were working together before merging with us. Branden and Yusaku used the same 5-folds based on a random split to train their level 0 models.

## Yusaku’s solution
I trained 10 level-0 LGBM models that utilized image features.
My models were largely based on the kernel https://www.kaggle.com/him4318/avito-lightgbm-with-ridge-feature-v-2-0/code (v14) with some tweaks and addition of CNN image features.
To add CNN features, the images were first resized to 224x224 and activations from the layer right before classification layers were extracted and average pooled.  The models were initialized with pre-trained weights from ImageNet.
The follow image features were used in the final models, though other CNN models were experimented with and did not work as well or were not usable due to memory constraints:

+ VGG16 (512x7x7)  ||  Average pool (512x1x1) - Improved 0.0008 LB

+ Densenet121 (1024x7x7) || Average pool (1024x1x1) - Improved 0.0012 LB

Both average and max pool were tried, but average pool gave better CV/LB, so I ended up just using average pooled features in the final models.
It would have been interesting to try lower level image features extracted from the CNN models (to better capture qualities such as image focus, fuzziness, graininess), but did not get around to it.

To encourage diversity and reduce correlations for the trained models, some were trained by removing features that were deemed important by LGBM, such as “image_top_1”, “city”, etc.  

For text features, TFIDF of bigrams were used, unmodified from the original base kernel.  I also experimented with adding text features based on gensim’s doc2vec but did not help.  I did not try FastText, but I probably should have based on favorable results reported by other teams.

My models’ predictions ended up having relatively low correlations with Branden’s models despite both of us using LGBM; combining my models with Branden’s gave a pretty big boost.
 
## Branden’s solution
He will release it in the comment field after few days!!!
