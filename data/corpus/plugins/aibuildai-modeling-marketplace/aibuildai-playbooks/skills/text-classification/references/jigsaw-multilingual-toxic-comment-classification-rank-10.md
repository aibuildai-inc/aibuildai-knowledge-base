# 10th Place Solution

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #10
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/161100

Thanks to my wonderful teamates @naivelamb @wuyhbb @terenceliu4444 @hughshaoqz , Thank you for helping me to get my 5th gold medal!

Congratulations to all winners!

# Summary

Here's main ideas that worked for us.

* Translated Data
* Pseudo Labelling
* Multi-Stage Training
* Freeze Embed Layer
* UDA (https://arxiv.org/pdf/1904.12848.pdf)
* Mono Language Modeling
* Test Time Augmentation

# Details

### Translated Data

We use translated data listed here for training and testing (for TTA)

https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/159888

Thank you guys for publishing this excellent datasets!

### Pseudo Labelling

We trained some baseline models then blend them with some public submission files to get to LB `0.9481` then use it as Pseudo Label till the end.

* https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta
* https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmroberta

When training on this test data with pseudo label, we found that use all test data by KL-Div loss with soft label gave us best performance.

### Multi-Stage Training

We all know that fine-tuning the model on validation set after training on train set can boost the LB score, and we call it `2-stage training`. (https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta)
By adding more stages on our training pipeline we were able to further boost our LB score. Some of our pipeline is like:

* Pseudo Labelling (5epo) -&gt; Train1 (1epo) -&gt; Valid (3epo)
* Train2 (1epo) -&gt; Train1 (1epo) -&gt; Valid (3epo)
* Train1 (1epo) -&gt; Pseudo Labelling (3epo) -&gt; Valid (3epo)
* ...

(Train1: training data of the 1st jigsaw competition on 2018)
(Train2: training data of the 2nd jigsaw competition on 2019)
(Pseudo Labelling: test data of this competition)
(Valid: validation data of this competition)

Note that we usually don't train on the full dataset when using Train1 and Train2, but a subset of it.



### Freeze Embed Layer

Freeze Embed Layer of transformers can save the GPU memory so that we were able to use bigger batchsize when training, while speeding up the training process.

By Multi-Stage Training, Pseudo Labelling and Freeze Embed Layer we were able to get to LB `0.9487` without blending any public submission file.


### Test Time Augmentation

We do the inference on all 6 languages, and the blending weight between the original language and the other 5 languages is 8:2
We got to LB `0.9492` by this TTA.


### UDA

reference: https://arxiv.org/pdf/1904.12848.pdf

Then we blend some models that used UDA while training to get to LB `0.9498`, by some further tuning we got to LB `0.9500`.


### Mono Language Modeling

At the last week, we found that blend with models that only trained on single language data can also boost the CV score.
But since in validation data we only got `es`, `it`, `tr`, so we only do `Mono Language Modeling` on these 3 languages.

By blending models that only trained on `es` or `it` or `tr`, we finally got to our best public LB `0.9504` and this is our best submission on private LB as well. 


---

# Thank you for reading!
