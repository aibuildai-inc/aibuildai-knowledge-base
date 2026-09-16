# 11th Place - GRU, CNN, Transformer - GPU Deep Learning!

Competition: child-mind-institute-detect-sleep-states
Rank: #11
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459596

What a fun competition! I enjoyed this competition! Thank you Kaggle and host.

# Building Blocks of GPU Deep Learning!
Below are 3 building blocks of deep learning. By creating different sequences of these building blocks, we can build powerful models! Each block takes an input shape of `(batch_size, sequence_length, features)` and outputs the same. The RNN "block" uses **bidirectional** (example code [here][6]). The CNN "block" uses sums and products of multiple **dilated convolutions** (from WaveNet paper [here][5] and example code [here][6]). And the Transformer blocks uses **self attention** (example code [here][6]). These 3 "blocks" can create features from signals without needing feature engineering!



# Data Loader - Cut Data into Daily Chunks
Our data loader will randomly give us 24 hour windows that **always contain both one onset and one wakeup**. During each epoch, we will get different cuts. This serves as crop data augmentation.
 

# Model Head is NLP QA Softmax (not NLP NER)
For the head of our model, we use an architecture that is **not** in any public discussion **nor** public notebook. We use a technique from NLP called QuestionAnswering (i.e. `BertForQuestionAnswering`). This is analogous to Computer Vision's `object detection`. All public discussions and notebooks use NLP NER (i.e. `BertForTokenClassification`) which is analogous to Computer Vision `segmentation`. 

The way to implement QA is for our data loader to always include one onset and one wakeup in every train sequence. Then we apply 1440 multiclass softmax loss to pick the correct minute of target. Later we will use a separately trained classification model to remove false positives (since our QA model was trained on 100% positives and doesn't locate negatives well).


# My CNN-Transformer-GRU, 80 features
**Approximately CV 0.800 LB 0.750**


# My Deep-GRU, 2 features
**Approximately CV 0.805 LB 0.755**


# Extra Predictions Trick
**Approximately CV +0.050 LB 0.050 Wow!**
A huge trick in this competition is to make multiple onset and wakeup predictions per night. First we make a first choice for every user every night. With 200 test users and approximately 30 nights per user and 2 targets, this is about 12000 predictions. Next we make a second guess (i.e. the second max probability) for each night and divide it by `2^1`. This is 12000 more guesses. Next we made a 3rd guess (i.e. 3rd max prob) and divide it by `2^2`. We continue this for 30 guesses per night where the last guesses are divided by `2^29`. This boost CV by about +0.050 wow 😀! 

This works because Average Precision metric is **always improved** when adding more predictions if the new predictions have a lower score than all previous predictions. (More explanation [here][1])

# Rerank Scores
**Approximately CV +0.020 LB 0.020**
For the metric Average Precision, the score we assign to each prediction is very important. First we use the probability from our above models as the starting score. Next we use CatBoost with handmade convolution features (**CatBoost achieves LB 0.705** by itself wow!), 1D-Unet, 2D-Unet, Mel Spectrograms, ResNet34, EfficientNetB5, and Audio WaveNet to rerank the scores. This is accomplished by predicting a new score for each prediction and then averaging all the scores.

# Ensemble with NMS and WBF
**Approximately CV +0.010 LB 0.010**
My CNN-Transformer-GRU and  Deep-GRU are very diverse. First I apply NMS (non maximum supression) to each model individually. Whenever two predictions are within 5 minutes of each other, we keep only the prediction which largest probability. Afterward we ensemble the two single models with WBF (weighted box fusion). Whenever two predictions are within 5 minutes of each other (and are the same level 1 thru 30 explained above in "extra scores trick"), then we replace the two with the average step position.

# Solution Code
I published the code for these two TensorFlow models [here][6]

[1]: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/discussion/229637
[2]: https://www.kaggle.com/code/ragnar123/wavenet-with-1-more-feature
[3]: https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-112
[4]: https://www.tensorflow.org/text/tutorials/transformer
[5]: https://arxiv.org/abs/1609.03499
[6]: https://www.kaggle.com/cdeotte/11th-place-gold-cv-835-public-lb-788
