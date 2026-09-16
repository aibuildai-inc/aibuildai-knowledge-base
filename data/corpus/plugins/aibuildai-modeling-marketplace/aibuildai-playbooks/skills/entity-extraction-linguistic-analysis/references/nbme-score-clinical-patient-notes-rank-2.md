# 2nd rank solution (inference code)

Competition: nbme-score-clinical-patient-notes
Rank: #2
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322893

I'd like to thank host and kaggle for this challenge. I learned a lot in this competition (foremost insight is that debertav3 has replaced roberta as the baseline for NLP models that can fit on a single GPU.)

I will describe my solution in detail today but I shared my best selected inference notebook: https://www.kaggle.com/code/cpmpml/sub-ensemble-010/notebook?scriptVersionId=94177123
It scored 0.893 on public LB (my best public score) and 0.894 on private LB.

It is 4 times 4 fold model model.  The 4 models are the same model with a choice of backbone (deberta-large or deberta-v3-large) and a choice of rnn (LSTM or GRU). Each model is trained on 4 folds and also 4 times on full train data.

**Edit:** My solution is described in this post: https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/323085
