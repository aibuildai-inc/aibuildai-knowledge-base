# 10th private (10th public) solution

Competition: liverpool-ion-switching
Rank: #10
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153696

We used 2 magic things in this comp:
1. Test was slightly shifted: we made it closer to the train distribution using cross-correlation;
2. Removing of 50Hz sine noise.

In our view, models were not so important in this competition but Wavenet variations performed slightly better than other models;
Our zoo is 6x Wavenets (cross-entropy &amp; focal losses), 2x CNN2d-Wavenet, 1x CNN2d, 1x LGBM.

Our best models (both CV &amp; LB) are Wavenet on 4 features (cleaned signal, signal difference, signal min &amp; max in the 22-window) &amp; CNN2d-Wavenet. 

We didn't use the noisy part of training data (part of 8th batch) and blended fold predictions (EMA-decay helped a lot!).

All our individual models (except LGBM) score gold on the private LB, but every blend performed better.

My teammates will describe some parts of the solution in the comment section.
