# 12th place solution

Competition: linking-writing-processes-to-writing-quality
Rank: #10
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/467328

### Acknowledgements

I would like to express my sincere gratitude to Kaggle and the competition host for providing an opportunity filled with learning and inspiration. I also extend my deep respect to all the participants who contributed their insights and ideas to the competition. Thank you.

### Overview

I trained lightgbm, xgboost, catboost, and denselight(from lightautoml) on two different sets of features, and averaged their predictions with different weights. The weight ratios were adjusted while observing the Local CV score and Public LB.

**Feature set 1**
1504 features, which include various features introduced by public notebooks,and some additional features. I added PCA, TF-IDF, CountVectorizer, and additional up_time_lagged features.

Reference public notebooks:
[enter-ing-the-timeseries-space-sec-3-new-aggs](https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs)
[silver-bullet-single-model-165-features](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features)

Feature set1 Lightgbm Result:
CV(10fold&5seed average): 0.596, Public LB:0.579, Private LB:0.571

**Feature set 2**
Features from public notebooks([silver-bullet-single-model-165-features](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features))and distilroberta predictions.

Feature set2 Lightgbm Result:
CV(10fold&5seed average): 0.589, Public LB:0.580, Private LB:0.570

**The weight ratio**
0.4 * (lightgbm(Feature Set 1) + 0.5 * xgboost(Feature Set 1) + 0.5 * catboost(Feature Set 1) + denselight(Feature Set 1)) / 3 + 0.6 * (lightgbm(Feature Set 2) + denselight(Feature Set 2)) / 2

Weighted Average Result:
CV(10fold&5seed average): 0.5870 ,Public LB: 0.576, Private LB: 0.563

### Training Strategy

For both feature sets 1 and 2, I trained all models with a 10-fold cross-validation on 5 different seeds and took the average. I performed parameter tuning for lightgbm and xgboost using optuna, while for catboost, I did not tune parameters and only set the depth to 6. For denselight, I used the parameters directly from this notebook. [lightautoml-nn-test](https://www.kaggle.com/code/alexryzhkov/lightautoml-nn-test)

### Feature set2's distilroberta
I used the reconstructed essay based on this notebook. [essay-contructor](https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor)

I performed preprocessing like the following: 
“qqq qq q’qq, qqq qqq” → “3 2 1’2, 3 3” 
Converted to the number of consecutive 'q’s.

Due to the vocabulary included in the tokenizer, it is only capable of encoding sequences of one or two characters, such as ‘q’ or ‘qq’, consecutively. Therefore, feeding it directly into the tokenizer results in an extremely large number of tokens. The following figure illustrates the change in the distribution of token counts before and after preprocessing.



Initially, I attempted to train using deberta-v3-large, but I could not achieve satisfactory result. When I switched to the lighter distilroberta, I obtained better results.The fact that there is limited data and the same token appears repeatedly may have led to better performance with lighter models.

deberta-v3-large CV(10fold&5seed average): 0.620 
distilroberta CV(10fold&5seed average): 0.605

However, it is likely that better accuracy could have been achieved with deberta-v3-base ,small or xsmall, as with many other top-tier solutions(deberta-v3-small is smaller than distilroberta). Since I started testing lighter models towards the end of the competition, I didn’t have time to consider a wide range of models and couldn’t investigate them. I am conducting follow-up tests with deberta-v3-base, small, and xsmall to see what results can be obtained. Once the results are out, I would like to share them in this form of an additional note.

To add diversity to the models, I only used the roberta features in feature set 2, since Feature set 1 already included many features extracted from reconstructed essay like TF-IDF. 

The improvement for the predictions of feature set 2 was as follows. 
| model | CV(w/o roberta )| CV(w/ roberta) | Public LB(w/o roberta) | Public LB(w/ roberta) | Private LB(w/o roberta)  | Private LB(w/ roberta) |
| --- | --- | --- | --- | --- | --- | --- | 
| lightgbm | 0.602 | 0.589 | 0.582 | 0.580 | 0.574 | 0.570|
| xgboost | 0.607 | 0.579 | 0.581 | 0.579 | 0.578 | 0.572 |
| catboost | 0.606 | 0.583 | 0.586 | 0.583 | 0.575 | 0.568 |
| denselight | 0.602 | 0.585 | 0.590 | 0.585 | 0.568 | 0.565 |

### What didn't work

dimension reduction(umap, tsne)
tsfresh
clustering
clasification
other loss function(huber, poisson ...)
wavenet
Inception time
Denoise auto encoder
Tabnet
MLP
autoint
ffttransformer

### late submission result (update 2024.01.24)

The deberta-xsmall model yielded the best results.
A token count of 1280 produced better outcomes than 512, and deberta was considered superior for its ability to increase the maximum token count significantly.
The presence of preprocessing led to better results. However, it should be noted that as the token count for deberta is increased, the difference becomes less significant.

**various  model result**
| model | CV | Public LB | Private LB |
| --- | --- | --- | --- | 
| deberta v3 xsmall max_token = 512 | 0.6088 | 0.6201 | 0.5999 | 
| deberta v3 xsmall max_token = 1280 | 0.5965	| 0.5879 | 0.6020 |
| deberta v3 small max_token = 512 | 0.6046 | 0.6152 | 0.5989 |
| deberta v3 xsmall max_token = 1280 | 0.5946	| 0.5994 | 0.5874 |
| deberta v3 base max_token = 512 | 0.6041 | 0.6135 | 0.5983 |
| deberta v3 base max_token = 1280 | 0.5955 | 0.5959 | 0.5906 |
| distil-roberta max_token = 512 | 0.605 | 0.6134 | 0.5900 |
| distil-roberta max_token = 512 w/o preprocess | 0.7638	| 0.7548 | 0.7820 |
| deberta v3 base max_token = 512 w/o preprocess | 0.7764 | 0.7991 | 0.7758 |
| deberta v3 base max_token = 1280 w/o preprocess | 0.6404 | 0.6392 | 0.6167 |

note: I used huggingface trainer. I modified commonlit public notebook. Parameter is same as reference notebook's parameter. [ref](https://www.kaggle.com/code/tsunotsuno/debertav3-w-prompt-title-question-fields)

**stacking result**
| model | CV | Public LB | Private LB |
| --- | --- | --- | --- | 
| deberta v3 xsmall max_token = 512 | 0.5878 | 0.5815 | 0.5717 | 
| deberta v3 xsmall max_token = 1280 | 0.5881 | 0.5789 | 0.5669 |
| deberta v3 small max_token = 512 | 0.5866 | 0.5820 | 0.5723 |
| deberta v3 xsmall max_token = 1280 | 0.5873 | 0.5820 | 0.5691 |
| deberta v3 base max_token = 512 | 0.5882 | 0.5800 | 0.5711 |
| deberta v3 base max_token = 1280 | 0.5862 | 0.5783 | 0.5703 |
| distil-roberta max_token = 512 | 0.5889 | 0.5804 | 0.5700 |
| distil-roberta max_token = 512 w/o preprocess | 0.5946 | 0.5850 | 0.5691 |
| deberta v3 base max_token = 512 w/o preprocess | 0.5965 | 0.5832 | 0.5730 |
| deberta v3 base max_token = 1280 w/o preprocess | 0.5972	| 0.5834 | 0.5729 |
| w/o bert predictions | 0.602 | 0.582 | 0.574 |

note: silver-bullet features + bert predictions, lightgbm and optuna, 10fold and 5seed
