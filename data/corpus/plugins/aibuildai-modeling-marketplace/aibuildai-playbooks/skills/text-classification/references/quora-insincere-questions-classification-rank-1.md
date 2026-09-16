# 1st place solution

Competition: quora-insincere-questions-classification
Rank: #1
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568

First of all, we want to thank Kaggle for hosting the competition and Quora for providing such a large dataset. Last 3 months were quite exhausting for us with a steep learning curve and tons of the ideas we wanted to try out. In the following we try to summarize some of the main points of our solution.

**Model Structure**
We played around with a variety of different model structures, but in the end resorted to a quite simple one that is very similar to those posted here https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/79824. It’s basically a Single Bi-LSTM 128 followed by a Conv1D with kernel size 1 only and GlobalMaxPooling afterwards plus additional dropout layers with minimal dropout. We additionally use a few statistical features.

![enter image description here][1]

**Embeddings**
First, we use all tokens from both train and test data for our vocabulary. We do the simple pre-cleaning that was posted in a kernel at the start of the competition and split by space afterwards (spacy and nltk resulted in similar performance). We do not lowercase, but keep uppercase, and do not limit the vocab at all. For embeddings we use glove and para where we weight glove a bit higher. The most important thing now is to find as many embeddings as possible for our vocabulary. We had a few steps to achieve this, like checking singular and plural of the word, checking lowercase embeddings, removing special tokens, etc. For public test data we had around 50k of vocab tokens we did not find in the embeddings afterwards. Even though we tried a few different strategies for handling the OOV tokens, we resorted to a single OOV token with a single random embedding vector. 

**Threshold**
We spent a lot of time trying to figure out good strategies for choosing a good threshold for classification. Over time, we saw that estimating the threshold on validation data and then applying it on test data does not really work. There is a large variation on optimal thresholds. So what we did instead is to try to find a fixed threshold on CV that produces the least deviation for the f1 score from the optimal threshold. We saw that we can get more stable results when we produce ranks on the predicted probability and average the ranks instead of averaging probabilities. For final submission we then chose the best CV threshold. This also allowed us to fit the model on the complete data without the need to rely on a random split and less training data. The visualization below shows that in action (not necessarily our final eval). On the x-axis we plot the different fixed thresholds and on the y-axis we see the deviation from the optimal F1 score across folds using this fixed threshold (see CV chapter below). The blue line is the mean, green is median, purple is minimum, red is maximum, and bars are stds. So for example here, if we choose a threshold in the range of 0.927 we expect the F1 score to be not much worse (around 0.001) compared to choosing the optimal threshold (which we can't do for test data). In practise, this might of course deviate further and we could also see larger deviations on PLB. For further elaboration, please check the comments.

![enter image description here][2]

**Runtime tricks**
We aimed at combining as many models as possible. To do this, we needed to improve runtime and the most important thing to achieve this was the following. We do not pad sequences to the same length based on the whole data, but just on a batch level. That means we conduct padding and truncation on the data generator level for each batch separately, so that length of the sentences in a batch can vary in size. Additionally, we further improved this by not truncating based on the length of the longest sequence in the batch, but based on the 95% percentile of lengths within the sequence. This improved runtime heavily and kept accuracy quite robust on single model level, and improved it by being able to average more models.

**Fitting**
We use a one cycle policy with Nadam optimizer (you can do this with the typical CyclicLearningRate implementations by just changing the step size to half your total iterations). We chose a batch size of 512. We could achieve similar results by even taking 10 or 20 times higher batch sizes, which goes hand in hand with recent research on fast convergence. With these larger batch sizes we could even fit close to 20 models, but results stabilized close to 10 models which is why we chose to go with the smaller batch size in the end. However, there might still be some room left here if one properly tunes this.

**Multiple models**
In the end, we managed to fit more than 10 models on the complete training dataset with help of the runtime tricks mentioned before. Our best final private score even had only a runtime of 6000 seconds (I think they used a bit better hardware for running), so there would be space for 1-2 more models. With larger batch sizes even much more might be feasible. As mentioned, we then average the rank predictions of each model and use our specified threshold for prediction.

**Embrace the randomness**
As it was necessary to utilize CUDNN Layers in this competition, there was some randomness involved that could be quite frustrating from time to time. I saw many people trying to fix seeds etc. and some claiming they could completely remove the randomness by using Pytorch (I still don’t believe this BTW as CUDNN has atomic operations). However, as mentioned before, a well working strategy in this competition was to combine multiple models and to end up with a good ensemble, those models should be a bit different to each other. So having different random initializations etc. can be helpful. Seeing people setting the seed as a hyperparameter is weird.

**CV Evaluation**
What I saw many people doing wrongly in this competition, and we also only figured this out after a while, is to trust their single out-of-fold evaluation. However, in this competition, it is crucial to combine (average) multiple models (in our case the same model). That means that our CV evaluation looks like the following. We do a k-fold split (mostly 10-fold) and fit the same model up to v-times on the same training split and then successively evaluate it on the single out of fold. So for the first split, we first fit one model and evaluate it, then a second one and evaluate the average and so forth. We repeat this for all 10 folds, landing us with e.g., 100 model fits overall, and then we can take a look at the median or mean over all folds for v-model-ensembles. The reason for doing this is that f1 scores are very different on the split you have. For one 10% split you might end up with a maximum of 0.72 and for the other you might end up at 0.705 or similar. So repeating the split 10 times, fitting the same model v-times for each split, and then looking at the grand picture gave us the best overall evaluation. This routine helped us to compare individual solutions with each other. BTW our final scores are exactly what we would expect from our CV evaluation, but again this might be lucky :)

**Robustness and over/underfitting**
Around 2 weeks before final submission, our results became so stable that changing things did not alter results much. Things like finding more OOV embedding vectors resulted in same results, using slightly different layers ended up being similar, and other things. This was a bit frustrating, but in the end things worked out. In the end, it was important to find a good balance between over and underfitting (as always). Underfitting too much led to good single model performances, but was worse for combining models, and the other way around. For example, if your model overfits, there can be many different solutions to tackle this, e.g., add dropouts, or reduce the vocab size, or reduce model complexity, etc. So if someone says on kaggle that one things works for him/her, that does not necessarily mean that it will work for you as you might already be doing something similar that has similar effects (a good example is the Gaussian noise discussion).

**What did not work for us**
Mostly you only read what worked, but here is an incomplete shortlist of what did not work for us. This does not mean that it doesn’t work at all, but rather that it was worse for our specific solution.

- Different optimizers (focal loss was similar though)
-  Label smoothing
-  Auxiliary learning / multitask learning
-  Snapshot learning
-  Pseudo labeling
-  Fitting own embeddings with gensim
-  Spelling correction
-  Taking median/percentile of predicitons instead of average
-  More complex layers and architectures (Attention, QRNN, Capsule, larger/multiple LSTM layers, larger CNN kernel sizes, LGBM or bag of words)
-  Word collocations - Several words put together can bear a completely new meaning, which is not captured by embeddings. Glove turned out to have quite a lot of such collocations with words put together using "-" sign. So we replaced examples like "ethnical cleansing" with "ethnical-cleansing", which is then captured by a more appropriate glove embedding. It showed no improvement on CV.
-  Extra statistical features - Presence of statistical features added a little bit to the accuracy based on CV, but we saw no improvement with other extra features, like sentiment or bag-of-words based variables.
-  Replacement of words with synonyms - An idea of replacing all nationalities (or e.g. political party) with the same word did not work at all.
-  Order the train data by the length of the sentences - This approach gave a dramatic improvement in the fitting time because each batch contained only sentences with similar sizes, but it hurt the accuracy of the model too much.


  [1]: https://i.imgur.com/zUY9tVN.png
  [2]: https://i.imgur.com/2NPwBIR.png
