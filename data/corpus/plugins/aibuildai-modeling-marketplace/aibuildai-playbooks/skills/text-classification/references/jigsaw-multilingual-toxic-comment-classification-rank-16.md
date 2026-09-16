# 16th Place Solution: Simple Calibrations

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #16
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160892

Congratulations to winner! 
Thanks to Kaggle team, organizers, educative and interesting notebooks ( @riblidezso , @shonenkov , @jazivxt ).
That was my first nlp competition, here I acquainted with SOTA models in NLP (GloVe/FastText + LSTMs -&gt; Transformer -&gt; BERT -&gt; XLM-ROBERTa, excited journey).
Take a gold medal will be too good to be true (I am the one who lost the most positions in top10), but anyway I glad to take my silver medal (the only one my medal 😃 ).

# Things that worked for me 
1. XLM-ROBERTa with LSTM + MAXPOOLING head lead to **0.9328 private** (0.9346 public)
2. Ensemble models from 1st step **0.9453 private** (0.9474 public)
3. Find bot comments by popular templates **0.9458 private** (0.9481 public)
4. Make simple calibration to the every lang in test set (convert pd from 3rd step to logit and then add some constants for every lang to logit, so it occurs that I need to add positive constant for fr and es, negative to it and pt), such additions give my final **0.9482 private** (0.9505 public) 

As I can see now I was overfitted from 2nd step on ensembles, didn't find good CV strategy
