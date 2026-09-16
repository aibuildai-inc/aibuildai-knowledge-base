# #3 solution | Many individual models and many ensembles

Competition: playground-series-s4e7
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s4e7/discussion/523661

First, I want to congratulate everyone who managed to handle this huge dataset and get on the board. No small feat. Next, I want to congratulate in particular to everyone in top 10, who stuck with this competition to the last day. There was quite a bit of LB shuffling in the last 5 days. Finally, I want to alert someone who placed below me that they will be getting a Kaggle t-shirt, as I am not eligible.

I was convinced from the beginning that all features should be treated as categoricals. I though initially that `Annual_Premium` had too many unique values (>53,000) to be used as such, and spent 4-5 days on getting that number down to 5,000-8,000. Obviously that didn't make it into day 1 solutions, and later @paddykb published a notebook showing there is no need to reduce the number of unique features in any variable. Most of my models dealt with data modified by `OrdinalEncoder` on each variable, resulting in the following tally of unique values per feature:

```
 Gender 2
 Age 66
 Driving_License 2
 Region_Code 54
 Previously_Insured 2
 Vehicle_Age 3
 Vehicle_Damage 2
 Annual_Premium 55068
 Policy_Sales_Channel 156
 Vintage 290
```

CatBoost worked best by far individually, but oddly enough not that well when ensembling models. For that purpose I used blending by numerical optimization (not really feasible with models > 10), Keras, LAMA NNs and LightGBM. The last two worked the best for ensembling.

If you look at my bio you will see that I am not a programmer. During each competition I create hundreds of scripts and run them locally, and none of them are integrated into a big pipeline. I like to divide everything into chunks and run things separately, which is why it is impossible to replicate my whole process here without a major effort. I will try to link the relevant notebooks already on Kaggle, and if I get a breather over the weekend I will try to publish at least one neural network. I will explain what I did, but you have a fair warning to stop reading here if you are interested only in code.

Here are my best 5 types of individual models based on CV scores:

| Model | CV | Public LB | Private LB |
| --- | --- | --- | --- |
| CatBoost Optuna | 0.896733 | 0.89728 | 0.89699 |
| Keras FM | 0.894276 | 0.89527 | 0.89498 |
| Keras embedding | 0.894192 | 0.89469 | 0.89445 |
| xLearn FFM | 0.893223 | 0.89447 | 0.89414 |
| LAMA ResNet | 0.893647 | 0.89378 | 0.89359 |

Of course I had many CatBoost models that were better than some models here, but this is shown only for variety. I also had 3 other types of LAMA NNs that I didn't use, also an xLearn FM model which I will explain below. Didn't try XGBoost at all. LightGBM was too slow with categorical variables and wasn't producing good models in my hands, but worked like a champion during the ensembling.

Others have already talked about CatBoost and LAMA, so I will focus on the middle 3 models. Keras FM refers to a neural network implementation of factorization machines. Rather than creating features by multiplying, dividing or somehow else combining features, we let  the NN do that for us. It is easy to Google factorization machines, so here is the gist. Those models convert all the features, numerical or otherwise, into factors/categories and explicitly model their interactions. They work really well for large datasets, especially in recommender systems, because of their speed. Since I converted all my features into categories, it was custom made for this type of analysis. Here is how the NN model looks like:

[Keras FM]

This will look tiny on the screen, and I suggest you right-hand click on the image and open it in a new tab, where you will be able to use a magnifier. It is a bunch of embedding nodes, one for each feature, that are crossed in dot-product fashion with each other. What that does is take two numerical vectors from embedding layers and squishes them together into a single number. Then there are linear representations of the same features, and finally everything is concatenated and passed through several dense layers. I also used dropout layers but they are not shown in that image. There is a link to an old script below describing Keras FM, and my implementation is similar except for the added dense layers.

https://www.kaggle.com/code/qqgeogor/keras-based-fm/script

Keras embedding was done similarly to [**this notebook**](https://www.kaggle.com/code/paddykb/ps-s4e7-keras-haz-insurance-losses) and I am grateful to @paddykb for the idea to convert all features into categoricals straight-up, without trying to reduce the number of bins. In my implementation Keras embedding didn't work great when all the features were categorical, but it added diversity. It worked better when autoencoder features were added as a separate layer to categoricals - see below.

xLearn FFM deals with field-aware factorization machines, which are similar in spirit to FMs but with a different type of data encoding. For an illustration how that looks like:

https://www.kaggle.com/code/ogrellier/libffm-model

I recommend that you try the xLearn package:

https://github.com/aksnzhy/xlearn

It works great with factorization models (both FMs and FFMs) and it is multithreaded - very fast. The latest version can't be installed by pip and isn't available on Kaggle, but I think the older versions should work fine. Most importantly, these models were extremely diverse with regard to everything else and contributed nicely to the ensemble, even though they were not great on their own. Below is an image showing cumulative distribution functions of the best CatBoost and xLearn FFM models. FFMs are much better at predicting 0s while CatBoost is much better at predicting 1s, and these two models complemented each other far better than any other two models I tested.

[CDFs]

Finally, I did some feature engineering by running a denoising autoencoder (DAE) and extracting its latent factors. I chose 3 and 8 factors at the bottleneck, and the latter worked better. Basically, we take the categorical data and convert them into a string of 0s and 1s using `pd.get_dummies`. Here I used numerical representations of `['Age', 'Annual_Premium', 'Vintage']` scaled to 0-1 range rather than their categoricals, as that would make a dataset very wide. These 8 features found by DAEs had decent predictive abilities on their own, as you can see in a t-SNE plot below.

[t-SNE autoencoder]

Still, DAE factors were not great alone, and worked the best when added to other features. That eliminated FMs because there were millions of unique values and couldn't be converted to categories, but individual CatBoost and Keras embedding models could handle these extra features and their scores jumped up by ~0.0002.

The final solution was a stack of 38 models made by LAMA DenseLight NN, but LightGBM had a near-identical solution. See [**here**](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/523405) for more details. There were at least 8 CatBoost models, some with and some without DAE features. Some models used the features from [**here**](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/516475). Also had 6-8 each of xLearn FM and FFM models, Keras FMs and Keras embedding models. I added 3 LAMA NN models at the very end and wish I had more of them, as they surely would have given a boost based on diversity with regard to other models. A couple of AutoGluon models were included as well. All of them were stacked together either using an Optuna-driven LightGBM, or as 10-fold Keras and LAMA NNs.

I want to thank everyone for excellent discussions, and for sharing your knowledge with patience that I sometimes lack. Special thanks to @paddykb and @ivanmitriakhin for publicizing the reversal of labels that gave most of us a nice LB boost.
