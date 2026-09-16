# From 400-ish public to 26 private

Competition: quora-insincere-questions-classification
Rank: #26
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80544

Hello, and congratz to everybody who made it 'till the end! Special thanks to people who shared stuff during the challenge, I learnt a lot.

I'll give you a brief overview of my model that made it top the 26th:

- Preprocessing : Some special characters cleaning, number processing, contractions &amp; mispells replacement and latex tags cleaning. No lowering though.

- Embeddings : Concatenation of glove, fasttext and paragram.

- Some features : Toxic words ratio, Total length, word vs unique words, ratio of capital letters.

**Model:** 

- I used PyTorch
- Single model , 5 folds, 4 epochs :
 - Embedding layer + some noise
 - LSTM, 64 Units (unidirectional)
 - GRU, 32 Units (unidirectional)
 - Attention, maxpool &amp;  average pool on the outputs of both rnns
 - Concatenating them with features
 - 32 units dense + reLu + Batchnorm + Dropout
 - And the final layer

CV : 0.688, Public LB : 0.700 

This model was not my best one on the LB, but it had a good CV and an average LB which made me trust it more than the others.

Thanks for reading, feel free to ask me any question! I'll probably make my code public, but it needs some cleaning first.
