# 8Th place solution

Competition: prostate-cancer-grade-assessment
Rank: #8
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169114

Despite some evidence of randomness I'd like to share the ideas we used:
* 10 model ensemble based on local CV and decent LB.
* Different predictions between models (regression, bins and ordinal regression)
* Some of it used bags of tiles and other stack the tiles in squares
* Efficient nets (I trained only b0 but partner had a few b4
* My models were trained in two steps. First a model with an attention layer is made (was shared by me in some thread). Then this attention layer and model is reused to predict weights for tiles. Then a model is retrained with a lower number of tiles (9 or 16). I have some 9 tiles models that were both fast and were going at 0.90CV+. On top of it it allowed us to inspect a larger amount of tiles during inference (128 tiles) and just select the best 9 or 16.
* My partner used in his model a NetVlad layer which maybe hell talk about in this thread.
* Ensembling with mean + round was better than majority voting for LB (and is what we used) but actually our best solution uses majority voting (which we didnt select).
* We also built a CV without duplicates and without "suspicious slides".

In the last weeks after making the team and how obvious it seemed the shake up would be big I started to mistrust LB and try to bring diversity to the ensemble. As long as a model was at LB &gt; 0.88 that was good enough if the CV was among the best ones.

Learned a lot during this competition. Thanks to organizer.

Note: We also have a solution at 0.936 that we didn't select :(
