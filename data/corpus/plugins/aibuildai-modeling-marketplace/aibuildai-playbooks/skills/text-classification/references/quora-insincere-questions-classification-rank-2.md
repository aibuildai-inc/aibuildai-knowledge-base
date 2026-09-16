# 2nd place solution

Competition: quora-insincere-questions-classification
Rank: #2
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/81137

## Summary
I used a single NN with 1-layer Bi-GRU and dense layers for statistical features. I did seed averaging for ensemble. I used PyTorch to write NN.

Key factors of my solution are

* Tune hyperparameters based on solid CV 
* Train word embeddings on the competition dataset. I guess most participants didn't?
* Faster training techniques to train more models. It's adaptive lengths of sequences to input RNN for each batch


## Preprocessing
I inserted spaces around characters except alphabets and numbers.
Then, I used keras tokenizer, which splits by only space.

After tokenization, I applied spell correction to OOV words. The rough idea of the spell correction algorithm is to find words with 0 or 1 levenshtein distance while ignoring cases. Precisely, there are a few heuristics.

In my case, devising preprocessing including the above spell correction did not change CV score so much.

## Model architecture
The main part is 1-layer Bi-GRU with hidden size 128 followed by the concatenation of max pooling, average pooling and first/last positional outputs. Another part is dense layers for statistical features. The outputs of 2 network parts are concatenated, then fed to dense layers.

<pre>QuoraModel(
  (embedding): Embedding(222910, 668, padding_idx=0)
  (text): RNNBlock(
    (rnn): GRU(668, 128, batch_first=True, bidirectional=True)
  )
  (features_dense): Sequential(
    (0): Linear(in_features=92, out_features=32, bias=True)
    (1): ReLU(inplace)
    (2): Linear(in_features=32, out_features=16, bias=True)
    (3): ReLU(inplace)
  )
  (dense): Sequential(
    (0): BatchNorm1d(1040, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.25)
    (3): Linear(in_features=1040, out_features=64, bias=True)
    (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): Dropout(p=0.1)
    (7): Linear(in_features=64, out_features=1, bias=True)
  )
)
</pre>

## Embedding
I used glove and wiki-news pretrained word embeddings. I also trained 64 dimensional word embeddings on the competition dataset (train+test) with fastText.
In addition to them, I added 4 binary features to word embeddings (all upper chars?, first char upper?, only first char upper?, OOV word?).
Finally, I concatenated all of them. The parameters of word embeddings are freezed during training.

## Statistical features
* the number of words
* the number of unique words
* the number of characters
* the number of upper characters
* Bag of characters: Implemented by `CountVectorizer(ngram_range=(1, 1), min_df=1e-4, token_pattern=r'\w+',analyzer='char')`

## Length of sequences to input RNN
For the faster training, I adjusted the lengths of sequences for each batch.
When training, I used the maximum length of sequences in the batch or 55 length by applying pre-truncation if the maximum length over 55. When predicting for test, the truncation is applied if the length is over 70 instead of 55.

Thanks to this trick, I was able to train 6 models on kernel compared 5 models without this trick.


## Training
I used Adam with learning rate 0.001. The learning rate is multiplied by 0.8 after each epoch.

I got the best CV score with batch size 256. But, batch size has the trade-off between score and training time. As I increase batch size, CV score gets worse and training gets faster. I chose batch size 320 by checking CV score and training time on kernel.

## Ensemble
I did seed averaging of 6 models. I trained 6 models with different seeds. 5 epochs are spent for each model. Each model is trained on the full train dataset, in other words, I didn't use k-fold split to train different models.

I averaged the predictions of 6 models. Then, I made the final binary predictions with threshold 0.36.

## Local validation
I did 5 fold CV for the local validation. For each fold, I used predictions after ensemble rather than predictions by 1 model for more stable CV, closer CV score to LB score and more optimal hyperparameter search when ensemble.

## CV score
I show CV scores of my model used for private LB and several models without some feature.
0.70974 is the CV score for the model used for private LB.

<pre>|Removed feature                   |score  |
|----------------------------------|-------|
|no removal                        |0.70974|
|4 binary embedding feature        |0.70957|
|spell correction                  |0.70953|
|statistical features              |0.70877|
|word embeddings trained on dataset|0.70794|
</pre>
