# #1 solution

Competition: tabular-playground-series-nov-2021
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291883

ambrosm & @pourchot congratulation and thank you for sharing some post processing ideas I was too lazy to investigate the chunk thing so I mostly rely on your kernel for that.

Beside that my solution was based on :
- training a simple NN and getting the oof on the train ~ LB 0.749
- from this oof, I have relabeled the top 5% mislabelled prediction and trained a simple NN with that new train ~LB 0.75010
- from this submission I have used 5% of the data as pseudo label and retrained a simple NN ~LB 0.75070
- blending this with @ambrosm & @pourchot kernel moved to my final LB

the only thing to modify is to use **rank **when you blend your prediction, since AUC is all about rank . Submission probability coming from different models are on different scale so it s certainly better to do something like 
sub['target'] = sub['target'].rank(pct=True)
before using them
