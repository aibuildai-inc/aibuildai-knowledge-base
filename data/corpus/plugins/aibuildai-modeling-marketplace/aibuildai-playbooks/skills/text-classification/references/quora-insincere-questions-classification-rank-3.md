# 3rd place kernel

Competition: quora-insincere-questions-classification
Rank: #3
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80495

Hi,
I have published the 3rd place kernel.
https://www.kaggle.com/wowfattie/3rd-place

I used a lot of others works. The key factors of my method are:
- Spacy tokenizer
- No truncation of tokens
- Try stemmer, lemmatizer, spell correcter, etc. to find word vectors
- 2 layer of globalmaxpooling
- checkpoint ensemble
- Local solid CV to tune all the hyperparameters

Questions, advises, suggestions are all welcome.

EDIT: I forgot to mention that all the punctuations are included.  "if token.pos_ is not "PUNCT" has no actual effect
