# [2nd Place Solution] from PPP

Competition: talkingdata-adtracking-fraud-detection
Rank: #2
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56328

Congrats to the top teams and thanks Kaggle and TalkingData for hosting such a perfect competition. Also congrats to Plantsgo, the new Kaggle grandmaster! 

Overall the competition was wonderful (except for D\*\*k's Kernel) and we learned much during the last month. Here I'd like to briefly describe our solution as well as some important techniques we used. To summarize, our solution consists of: 

- a framework that is both time and memory efficient to cope with the large dataset,

- some regular features,

- two methods, LightGBM and NN,

- a simple weighted average ensemble of predictions.

## Framework
It is essential for us to use sub-sampling to reduce time and memory costs. At the early stage, we found it so hard to deal with the whole dataset due to the limitation of RAM (128G for Plantsgo, 64G for me, and 32G for Piupiu). Piupiu bought another 16G RAM immediately, but he was upset when finding that it merely helps. The difficulties to deal with such large data are 2-folds: it's hard to extract features, and it's slow to train a model. As the training data is extremely imbalanced, we reduced the data size by sampling a small fraction (5%) of negative samples. In the feature engineering phase, new features were extracted from the entire data and merged into the sub-sampled data. In the training phase, only the sub-sampled training samples were used so it was about 10+ times faster than directly training the entire data. We used 5-fold CV to see the offline performances. It took about half an hour to train one fold (with an Intel i7 core and two hundred of features).

## Feature engineering
Our team did not have any magic features, although my teammates Plantsgo and Piupiu are both feature engineering experts. Our features were regular in the sense that almost all our features were open-sourced by others in the Kernels one or two weeks after we had used them. What a sad story. There were several kind of features: count features, cumcount features, time-delta features, unique-count features, and which [app/os/channel]s each IP appears in the data. Because we have the sub-sampling framework, we have enough memory space to get hundreds of features.

## Models
We used two methods, LightGBM and NN. The best single model was LightGBM with Plantsgo's features which scored 0.9837 on the private LB. I have been trying to make a strong neural network to beat LightGBM during the whole month, but obviously I failed. Our best NN scored 0.9834 on the private LB which had a dot-product layer for categorical inputs and deep fully-connected layers for continuous numerical inputs. I believe there must be better NN structures and I really hope to learn it from other top teams. 

## Ensemble
The three of us had three LGB predictions and three NN predictions. So we averaged these 6 predictions by trivially applying some weights inferred from their public LB scores. Now that the private scores are revealed, we find it more correlated to the offline CV scores rather than the public scores. If we had trusted the offline scores, our final score could have been better. 

### The leak on the test set: a sad story
In a word, we did not use the leak on test set, even though it was us to post the topic to Discussion. We were filled with grief when we heard that the leak would help improve about 0.0004. 

Thanks for reading it! 

也谢谢大家的支持！
