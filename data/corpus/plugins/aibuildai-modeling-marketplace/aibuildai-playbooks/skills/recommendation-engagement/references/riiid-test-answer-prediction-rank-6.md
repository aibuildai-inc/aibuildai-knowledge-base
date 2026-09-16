# 6th Place Solution: Very Custom GRU

Competition: riiid-test-answer-prediction
Rank: #6
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209581

First of all congrats to @keetar and @mamasinkgs and other top teams. I will shortly explain my solution. It is late here, so I may be missing some points.

**Some General Details**

- Used questions only. Lectures improve validation score but increase train-val gap and don’t improve LB.
- Didn't use dataframes. Used numpy arrays partitioned by user ids.
- Sequence length 256
- 15 epochs. 1/3 of the data each time with reducing LR.
- 8192 batch size
- Ensemble of the same model with 7 different seeds trained on whole data
- 0.8136 single model validation score, 0.813 LB. Ensemble: 0.815.
- 8 hours training on 4-GPU machine
- Used Github and committed any improvement with a message like: Add one more GRU layer (Val: 0.8136, LB: 0.813)

**Inputs**


**Engineered Features**

Assume current question’s correct answer is X. Logarithm of:
- Number of questions since last X.
- Length of current X streak. (can be zero)
- Length of current streak on any non-X answer. (can be zero)

This helps with users who always pick A as answer etc.

**Embeddings**




Content Cosine Similarity
- A bit similar to attention with 16 heads
- Linear transformation and l2 norm applied on content vectors
- For 16 different transformation, cosine similarity between current content and history contents are calculated.
- Transformation is symmetric for content and history contents.

U-GRU:
- GRU with 2 directions but not BiGRU
- First does reverse pass, concatenates the output and then does forward pass

MLP:
- 2 layers of [Linear, BatchNorm, Relu]


Edit: Part embedding is trainable. There is actually sigmoid x tanh layer before GRUs.
