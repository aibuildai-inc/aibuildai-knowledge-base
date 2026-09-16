# 7th place solution - bucketing

Competition: quora-insincere-questions-classification
Rank: #7
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80561

Hi,

Here I explain my solution.

I think bucketing and checkpoint-ensembling are the key factors of my solution, since my preprocessing and my model are quite basic.

# Preprocess
The core part here is using NLTK TweetTokenizer.

1. Split each question_text by " " (space).
1. Replace words that has "\*" with "FWORD", since NLTK TweetTokenizer will split by "*", but I want to use words like "f\*\*k" as the single token rather than ["f", "\*", "\*", "k"].
1. Join by " ", then apply NLTK TweetTokenizer.
1. Split each word by ' (single quote) and -. e.g., ["it's", "nice"] -&gt; ["it", "'s", "nice"]
1. Load pretrained embeddings. For "FWORD", using the average of the embeddings of ["fuck", "shit", "\*"]. For OOV, using the average of embeddings.

I think that the preprocessing other than applying TweetTokenizer doesn't make big difference, since whether applying such "*"-replacement or not doesn't change the local CV score. The only reason why I subimitted this version is just I couldn't ignore the time I spent for preprocessing. XP

# Model

1.  Embedding layer. Simple average of Glove and Paragram embeddings (thus dim=300). Keep fixed.
1. Dropout (keep_prob=0.6)
1. Bi-LSTM (each cell_size=128)
1. Bi-LSTM (each cell_size=128)
1. Concatenation of the average-pooling of the first Bi-LSTM, the max-pooling of the second Bi-LSTM and attention of the second Bi-LSTM. (thus dim=3\*256)
1. Dense with tanh (dim=32)
1. Output with sigmoid

# Training
## Use bucketing.
Bucketing is to make a minibatch from instances that have simillar lengths to alleviate the cost of padding. This makes the training speed more than 3x faster and thus I can run 9 epochs for each split of 5-fold.

I must have seen the TensorFlow tutorial page that describes bucketing (it shoud be the tutorial of "sequence-to-sequence model"), however, somehow I couldn't find that page now.

For other training details,

* Objective function: vanilla sigmoid\_cross\_entropy
* Optimizer: Adam with default parameters
* Batch size: 512
* Maximum sequence length of the input: 400

# Postprocess
For each 5-fold model, apply checkpoint-ensembling to maximize each validation score.
Without checkpoint-ensembling, the average validation score is about 0.694. After checkpoint-ensembling, it is about 0.700.

After checkpoint-ensembling, ensemble 5 models by averaging output probabilities and thresholds, then submit.

Finally, Thanks everyone working for this competition! I really enjoyed this competition with such a big data!
