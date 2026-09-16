# Public 16th / Private 10th Solution

Competition: shopee-product-matching
Rank: #10
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238039

Thanks to all the participants for their hard work. I am new to this field and only scored about 0.7 for the first 50sub or so. However, by studying from similar competitions and papers in the past, I was able to win the gold medal ! This competition was also a good opportunity for me to learn about engineering because of the tight limits on inference time and memory.

# Overview

[shopee]

# Details

## image models

First, I cleaned up the labels, since images that look almost identical can become noisy when used to train an image model. In other words, I re-labeled labels that are higher than a certain level of similarity as the same label.  
Then, I trained each models by ArcFace and extracted the embeddings.

## title models

After processing the escape string, I trained `indobenchmark/indobert-base-p2`, `indobenchmark/indobert-large-p2` and TFIDF. `indobenchmark/indobert-base-p2` was better than `bert-base-uncased`.

## Query Expantion

This competition guaranteed that there would be more than one label for each product.
Therefore, the αQE with n=2 and normalized similarity was very effective. 

## MetaModel

After training the above models and QE, I trained lightGBM model by similarity feature in the pairwise dataset.
However, trying to predict LightGBM normally on the kernel is very time consuming, so I used cuml's ForestInference to predict it. That was insanely fast.
Finally, I reapplied αQE by using prediction of lightGBM (so-called Discriminative QE). 
As it turns out, this DQE wasn't very useful.

## Threshold searching

First, I used only similarity of abcde' embdding, the score was relatively low. (But abcde' embdding is high CV)
So, I explored this cause. In the image below, you can see that some of the predictions of abcde'(cnn_bert_distance) are not as confident as when using abc' (cnn_all_distance). 
I combined multiple predictions by carefully observing the scatter plot in this way.
[image]

Also, by experimenting with local, I confirmed that tuning the threshold in public does not affect the results in private, so we adjusted the threshold in public.

## Didn't work

- Some diffusion technique.
- [Attention-based query expantion](https://filipradenovic.github.io/publications/grb20.pdf).
- MLP trained by embedding based pairwise dataset.
- Unsupervised model (like CLIP)
- Arcface with triplet loss

---------------

## Late Submission

I didn't try it because I didn't have time to do it since I came up with the idea of using LightGBM about a day before yesterday, but I got the following scores with LightGBM only prediction (lgbm > 0.8). So, all I need is metamodel.
[late sub]
