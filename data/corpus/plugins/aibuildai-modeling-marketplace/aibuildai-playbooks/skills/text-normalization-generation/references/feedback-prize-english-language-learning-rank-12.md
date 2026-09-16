# 12th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #12
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369565

First of all I would like to thank the competition organizers and Kaggle and congratulate all the winners.
I would also like to thank all the kagglers who provided useful information in the various discussions.
I have been working hard to achieve Solo Gold and am happy to finally win it.

# Overview
My solution is an ensemble of various deberta models and RAPIDS SVR.
I have been referring to @cdeotte excellent notebook on RAPIDS SVR. Thank you very much.
[https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x](url)

# Preprocessing
Two types of text data were prepared, one with \n converted to [BR] and the other without.
The model trained on the text data converted to [BR] scored higher for both Public and Private.

# Cross validation
I used 5folds in MultiLabelStratifiedKFold by target.

# Model
backborn
- deberta-v3-base
- deberta-v3-large
- deberta-large
- deberta-xlarge
- deberta-xlarge-mnli

pooling
- meanpolling&layernorm

# RAPIDS SVR
Features were extracted from the following models and SVR trained.
- deberta-v3-base
- deberta-large
- deberta-v3-large
- deberta-xlarge
- deberta-base-mnli
- deberta-large-mnli
- deberta-xlarge-mnli
- deberta-v2-xlarge
- deberta-v2-xlarge-mnli

# ensemble
Weights were adjusted by target using the Nelder-Mead method.

# What worked
- Soft Labeling
- Pseudo Labeling
- Small batch size (batchsize=2)

# What didn't work
- MLM
- Non-deberta models

# What worked in Private
The ensemble with GPR (Gaussian process regression) had the highest private score, but was not selected because the public score was not high.(public:0.437852/private:0.433646)

# Efficiency Prize(4th place)
- deberta-v3-xsmall
- Soft labeling
- Pseudo Labeling
- Use the model with the highest CV score out of 5folds
