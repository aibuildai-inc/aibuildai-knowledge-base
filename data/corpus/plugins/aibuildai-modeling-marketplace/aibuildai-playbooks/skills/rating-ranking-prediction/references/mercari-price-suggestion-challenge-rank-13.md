# 13th Solution

Competition: mercari-price-suggestion-challenge
Rank: #13
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50260

Kernel : https://www.kaggle.com/muhammadalfiansyah/push-the-lgbm-v19/code

This model is ensamble of ridge RNN and LGBM. 

I train ridge and RNN on 2 fold then use their prediction as input for my LGBM model.

Ridge : Mostly 2-3 ngram from name and description. I also train another ridge with transformed target.

RNN :  https://www.kaggle.com/nvhbk16k53/associated-model-rnn-ridge

Nothing special I add. I just modify to use only one dense layer, smaller GRU, smaller sequence length to speed up the training process. 

LGBM : https://www.kaggle.com/muhammadalfiansyah/simple-lgbm-2500-boost

I found out that LGBM workbest only using 1 ngram and countvectorizer. adding more ngram will degrade the performance. 
I also found out the more I overfit my NN the better the ensemble score is.
I found out that better score for each individual model doesn't mean better ensamble score. 

Ridge Only = 0.48xx

LGBM Only = 0.419x

RNN Only = 0.44xx


Ridge OOF+ LGBM = ~0.407

RNN OOF+ LGBM = ~0.408

Ridge OOF+ RNN OOF+ LGBM = 0.400x

I think the key in my ensemble is having different set of input for each model.



Hope that helps!
