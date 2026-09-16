# 9th place short summary

Competition: liverpool-ion-switching
Rank: #9
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153712

Short summary of our solution. 

1) 'magic' shift 2.724 for 11class batches and deleting of 7th batch from train. It was our first "discovery" in this competition

2) combination&nbsp;of some wiener filters and bandstop filter for signal cleaning. It helped a lot for simple models (GMM 0.941 on CV, Naive Bayes 0.942 on CV) but unfortunately it had no any effect for our wavenet. So we just used it for our GMM and afterwards GMM weighted prediction and GMM probabilities were used as features for our NN

3) cleaning of 50hz noise

What we've tried but it didn't help:

1) HMM sampling
2) blending/stacking
3) TCNN
4) CRF
