# 1st place solution

Competition: mercari-price-suggestion-challenge
Rank: #1
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50256

Hello everyone, thanks for an amazing competition. Together with Konstantin we are very excited, probably too much to describe our solution in greater detail - tomorrow we will write more. 

However we have something to whet your appetites. Konstantin did an amazing job and created a very small kernel based on our work over those 3 months that achieves 0.3875 on the validation ... and it is only 83 LOC - how cool is that!

https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s

UPDATE 2

Link to solution https://github.com/pjankiewicz/mercari-solution

~~~~~~~~~ UPDATE ~~~~~~~~~

Full description 
----------------------------------------------------

Pawel’s premerge solution (~0.3950)
-----------------------------------

I had a pretty complicated models before teaming up with Konstantin. It consisted of 3 parts:

1. Models per category - I created ridge models for category level 1, 2, 3, combinations of category 2 + shipping and item_condition_id etc. They were quite fast to train and created 0.4050 solution in about 20 minutes.
2. Residual models MLP - based on the 1. models I trained next models on sparse inputs (neural networks). The target was the difference between the predictions and real prices (think of a boosting but with strong models).
3. Residual model LGBM - the same idea as 2. 

While the concept of training models per category seems like a good one it is really not. Training models like this is cool but it underestimates the model one is using. You can think that a neural network can’t be trained to understand the interaction between categories and descriptions, brands etc. This is a wrong assumption. Our example shows that a single well tuned neural network can learn what it wants.

Konstantin’s premerge solution. (~0.3920)
-----------------------------------------

Before the merge, I had two base models:

1. Sparse MLP implemented in Tensorflow, without any sophisticated features, compared to public kernels I tweaked only the number of features, and tokenizer using eli5 explanations of the Ridge model. 3 models were trained sequentially.
2. CNN with conv1d, similar to what was in many kernels, implemented in Keras. The model itself was not great, but it was very different from an MLP model, so gave a nice boost.

The merge
----------------

At the moment of merging Konstantin was first and I was second. Someone would say “why did we merge at all”. But we did and it was a good decision. We expected that in order to succeed it wasn’t enough to blindly merge 2 solutions. In fact we had to create 1 solution which uses ideas of both of us. Making it work under the constraints was very challenging. The time wasn’t such a big deal for us, we focused more on the memory and being able to run 4 neural networks at the same time.

It took us maybe 2 weeks to come up with a unified solution (probably we worked the hardest at that point). It turned out that 2 different preprocessing schemes (2 datasets) created much needed variance in the solutions and we easily improved by 0.01 which was quite a big jump.

In the end we used 3 datasets and 4 models in each dataset. We tried to diversify the models:

- different tokenization, with/without stemming
- countvectorizer / tfidfvectorizer

Build system
------------

It was quite impossible to manage this project without a way to split our solution into modules. We independently created a build system for our code. Eventually we used the system that created a Python package with the solution. Our script was strange and looked like this (pseudocode):

    ENCODED_FILES = {some base64 encoded characters}
    DECODE_FILES 
    INSTALL THE MERCARI PACKAGE
    RUN MAIN FUNCTIONS

It was really important to run models in 3 sequential processes because Python has a tendency not to clean the memory after some operations (especially data preprocessing).

Feature preprocessing
------------

Some tricks/no-tricks that worked:

- name chargrams - We don’t know why exactly but using character n-grams from name improved the score. Maybe it was because it produced relatively dense features
- stemming - we used a standard PorterStemmer
- numerical vectorization - we noticed that a very big source of errors were bundle items with descriptions like: “10 data 5 scientists” were vectorized to data=10, scientists=5. This vectorizer applied in only 1 dataset improved the ensemble by 0.001. We didn’t have much time to test this idea further. 
- text concatenation - to reduce the dimensionality of the text fields by just concatenating them together - we tested all configurations {name, item_description, category, brand}. This was a reason for the 0.37xx push. 

Whatever cool idea we had about additional feature engineering didn’t work. To name a few:

 - Extraction of features like “for [Name]”. We noticed that many items were designated to a particular person. We weren’t sure what it meant exactly but it seemed important enough to create a feature. We created a list of names from nltk and searched for similar strings with AhoCorasick algorithm.
- We noticed that there were issues with new lines in descriptions. Wherever someone used a newline in description it concatenated the words likethis.
- Spell checking.

Quoting Pawel:
neural networks are like "ok I guess I can use your _feature engineering_ here you are 0.0003 increase"

Models
--------------------

Pawel discovered that with a properly tuned sparse MLP model, it’s better to achieve diversity in the ensemble by training the same model on different datasets, compared to different models on the same dataset. So we decided to stick to one model, which really simplified our solution. The base model is a multilayer feedforward neural network with sparse inputs: nothing fancy at all, so it’s surprising it works so well. I think one reason is that it’s more efficient compared to other approaches: the dataset is large, so the model must have big capacity and be able to capture feature interactions. But in the time that a conv1d CNN could train on 4 cores with embedding size 32, we could train 4 MLP models with hidden size 256.

Careful learning schedule tuning is crucial: most importantly, we doubled batch size after each epoch, which made the model train faster after each epoch and also lead to better final performance. We also lowered the learning rate in addition to batch size increase. Everything was tuned to get the best validation score after the second epoch, and then overfit after the third epoch: this made the model much stronger in the ensemble.

Compared to the model in the kernel (https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s), we had some extra tweaks:

- We had two model variations that we used: one was trained with Huber loss (it’s better here as it’s less sensitive to outliers), and another was doing classification instead of regression. For classification, we split all prices into 64 buckets, and created a soft target for prediction: first calculate L2 distance from the centers of the buckets, and then apply softmax to this (with high softmax temperature which was a hyperparameter). This classification model achieved better score on it’s own due to less overfitting, and also added diversity.
- For 2 out of 4 models for each dataset, we binarized input data during training and prediction by setting all non-zero values to 1. This is like getting an extra dataset with a binary CountVectorizer instead of TFIDF for free. I think it’s a great idea, kudos to Pawel! We also tried to binarize using non-zero thresholds, but this didn’t bring much improvement.
- L2 regularization helped a little (we applied it only to the first layer), also Tensorflow models worked better with PRELU activations instead of RELU.

Model implementation and optimization: as it’s a kernel competition with very hard constraints for this dataset size, it was important to make training maximally efficient: by making the first hidden layer bigger, it was possible to improve the score, and we also had 12 models to train in addition to preparing dataset. In the end, we managed to train 12 models with hidden size 256 on about 200k features. Here is how we did it:

- TF can use multiple cores, but it’s very far from linear scaling, especially for sparse models. So it is better to use one core for each model and train 4 models in parallel: we restricted TF to one core with OMP_NUM_THREADS=1 and usual TF config variables, and then allowed TF to use multiple cores with threading using undocumented use_per_session_threads=1: this meant that we didn’t need to start multiple processes and used less memory.
- It turns out that MXNet allows for a more efficient CPU implementation of a sparse MLP, because it has support for sparse updates (maybe TF has it too, but we were not able to use it). Pawel wrote an initial MXNet version that was about 2x faster, and then we added all features from TF models and made it work even better. The problem was that MXNet executor engine is not thread safe, if you try to parallelize using threading you get either 1 core used, or segfault. So we had to move to multiprocessing and to also write our own data generator because MXNet one made a lot of copies and used too much memory. We also had a version with putting data into shared memory, but it was way too close to disk space limit, so we had to scratch that.

Overall MXNet solution was faster and allowed to use a smaller initial batch size without loss in speed, but used more memory and looked less reliable. So at the end we had two submissions using the same datasets, one with MXNet (0.37758 private LB / 0.37665 public), and one with TF (0.38006 private / 0.37920 public).

At the end, we had 12 predictions and needed to merge them. An average worked well, but it was better to tune blending weights, so we used 1% of the dataset for validation (5% locally) and tuned the weights using a Lasso model. Lasso uses L1 regularization, so it was a bit funny when you have some idea and add more models, but Lasso says: meh, I don’t want your new models, and sets their weights to zero.

Models that did not work:

- MoE (mixture of experts): this is a really cool paper https://arxiv.org/abs/1701.06538 which describes how to train a model which has more capacity while using the same compute budget: exactly what we want here! But at the end it turned out that TF lacks sparse support for some operations required to make it work.
- We tried to add some exotic model blends for example: merging FM and MLP in the same architecture, adding a skip layer with linear regression to the output of MLP. It all eventually converged to a simple MLP. 

We achieved a point where only ways to improve was to train bigger MLP networks but due to constraints we couldn’t. This was really evident because models improved only when we added more data/features to the ensemble.

Meta Statistics
-------------------------

- Slack messages: 4500
- Git commits: 152 (317 across all branches)
- PR merged: 31
- Python LOC: 2015
