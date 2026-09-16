# 4th place solution (with github)

Competition: quora-insincere-questions-classification
Rank: #4
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/81632

Hi guys,  
It is a little bit late, but I published my solution as below:  
https://github.com/k-fujikawa/Kaggle-Quora-Insincere-Questions-Classification  
https://www.kaggle.com/kfujikawa/4th-place  
Here I will try to summarize some of the main points of my solution.

# Summary

The key factors of my solution are:

- Word2Vec fine-tuning
- 400dim random sampling from 600dim word embedding per CV
- Simple 2layer BiLSTM model with maxpooling
- 5-fold CV and averaging model outputs

[overview]

# Details

## Preprocessing

I refered to the public kernel (https://www.kaggle.com/hengzheng/pytorch-starter
) for the most part, and I made slight modifications as below:

- Exclude filter of punctuations that [Keras Tokenizer has by default](https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py#L169)
- Apply misspell corrections before punctuation spacing
- Insert spaces around characters except alphabets and numbers

## Embedding

In order to improve the word embeddings which are frequent in Quora dataset but not included in pretrained vectors (Glove and Paragram), I fine-tuned the word embeddings on the competition dataset (train+test) with Word2Vec (CBOW).
I show the results of preliminary experiments to confirm whether these word embeddings are improved or not.  
https://www.kaggle.com/kfujikawa/word2vec-fine-tuning

I attempted to use word vectors obtained by concatenating before and after fine-tuning, but it was difficult due to the problem of calculation cost.
Therefore, I decided to obtain word embeddings from 600 to 400 dimensions randomly for each CV.
This approach was effective not only to reduce computational cost but also to increase model diversity among CVs, so contributed to improve the score of the Public LB, although the score of the local CV has decreased.

## Model architecture

I adopted simple 2layer BiLSTM model with maxpooling.
Model details are shown as below:

    BinaryClassifier(
      (embedding): Embedding(
        (module): Embedding(212418, 402)
        (dropout1d): Dropout(p=0.2)
      )
      (encoder): Encoder(
        (module): LSTMEncoder(
          (rnns): ModuleList(
            (0): LSTM(402, 128, batch_first=True, bidirectional=True)
            (1): LSTM(256, 128, batch_first=True, bidirectional=True)
          )
        )
      )
      (aggregator): Aggregator(
        (module): MaxPoolingAggregator()
      )
      (mlp): MLP(
        (layers): Sequential(
          (0): Linear(in_features=262, out_features=128, bias=True)
          (1): ReLU(inplace)
          (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): Linear(in_features=128, out_features=128, bias=True)
          (4): ReLU(inplace)
          (5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (out): Linear(in_features=128, out_features=1, bias=True)
      (lossfunc): BCEWithLogitsLoss()
    )

## Statistical features for words

- Whether or not the word is included in pretrained embedding
- IDF score

## Statistical features for sentences

- the number of characters
- the number of upper characters
- the rate of upper characters
- the number of words
- the number of unique words
- the rate of unique words
