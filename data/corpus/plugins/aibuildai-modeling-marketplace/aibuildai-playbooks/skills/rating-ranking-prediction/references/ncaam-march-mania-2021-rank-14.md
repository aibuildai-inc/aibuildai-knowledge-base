# 14th place solution

Competition: ncaam-march-mania-2021
Rank: #14
Source: https://www.kaggle.com/c/ncaam-march-mania-2021/discussion/230929

First, congrats to the new champions Baylor!

My solution was based on 4 models, as like many others here from top models (3) from previous NCAA competitions with some minor changes. All credit to them.
https://www.kaggle.com/scirpus/jeepy-2021
https://www.kaggle.com/jonbown/creating-sequential-data-for-random-forest-in-r
https://www.kaggle.com/imoore/2019m-1st-solution-with-parameter-optimization

And 1 own model, a Stacking regressor with 10 regressors models AdaB, Xgb,LGBM, CatB, RFR, LR, RidgeCV, GBR, HGBR and LSVR. 

All models were then validated and tuned against this dataset and previous competitions. Based on the validation I used a weighted moving target, shifting predictions based on the four model’s outcome, from worst to the best validation, hopefully towards the direction of the better predicted value. 
I didn’t use any manual changes in the prediction and model also had Baylor as the final champion 😊

This competition was my first in the March Machine Learning Mania series and it has been a roller coaster, but really fun watching and following the tournament 😊 I take many notes with me to the next years version, like not using too aggressive approach, had the 1st place LB position in NCAAW but lost 1 game with probability 1.0, and the LB position went down you can say 😉

Now I’m glad and honor for this 14th place and silver medal, and a bronze medal in the NCAAW. This also makes me the highest ranked Kaggle Expert with total rank of 36 😉 always something!, wonder if there is a special prize/merchandise for it? ;) but Kaggle Master is in the target, someday in some competition.

Thanks for the reading the mini writeup!
And Thanks to Kaggle and NCAA for this great tournament!
