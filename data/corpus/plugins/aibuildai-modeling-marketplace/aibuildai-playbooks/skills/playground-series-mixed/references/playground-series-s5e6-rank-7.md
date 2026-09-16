# 7th place solution - HC + Ridge

Competition: playground-series-s5e6
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s5e6/discussion/587573

Thank you for the past month!
It was an honor to achieve this ranking in a competition I participated in wholeheartedly from start to finish.
In every competition, I make it a point to explore and deeply experiment with new techniques.
In this competition, I challenged myself with multi-level ensembling, something I had never tried before — up until now, I had only done simple weighted averaging of multiple models at most.
Below, I will share my 1. model,  2. reflections, 3. acknowledgments.

## 1. model

###### L1
For the first 20 days or so, I focused on implementing base models for ensembling.
To increase diversity, I varied the following 3 aspects, tuned the remaining hyperparameters for optimal performance, and kept only the models that contributed to improving the overall prediction.
1. how many original datas added to the training data
1. treat numerical features as cat or int
1. reg_alpha, reg_lambda params

As a result, 7 XGB models(→RIDGE1) and 6 LGBM models(→RIDGE2) remained.
I also made NN model (refer to [here](https://www.kaggle.com/code/ricopue/s5e6-fertilizers-nn-keras-all-embedding), changed some points like num of folds).
###### L2
I used 2 public codes( [one](https://www.kaggle.com/code/ravaghi/comparing-multiclass-ensembling-techniques) [two](https://www.kaggle.com/code/ayushchandramaurya/predicting-fertilizer-name-stacking-ensemble) ) in order to improve the score.
I pruned the ensemble candidates that used approaches different from mine as much as possible.
###### L3
At this layer, I ultimately used both HC and RIDGE.
I also experimented with stacking models like NN and XGB, but they didn’t help improve the score at all, so I ended up abandoning them.
Although I was initially hesitant to use HC at this layer, the cross-validation score for the final submission was clearly better, so I decided to go with it.
###### L4
HC(L3)
OOF : 0.38396, private LB : 0.38460
RIDGE(L3)
CV : 0.38368, private LB : 0.38449
→last submission(L4)
CV : 0.38412, private LB : 0.38486

## 2. reflection
###### positive aspect
Reflecting on the previous competition and heeding the warnings from Discussions, I was able to fully trust my CV this time.
To be honest, since I’m still relatively new to Kaggle, I used to get overly excited or discouraged by the public LB, but in the final week, I managed to stay sane :)
###### negative aspect
First of all, I completely forgot that ridge regression has a hyperparameter.
It happened to work well with the initial setting, so I left it as is and got caught up in building the stacking model.
I only realized this just now after looking at other people’s solutions…

Also, I had been naming all my submission files the same (ensemble_submission.csv), so I lost track of which submission was which.
As a result, I ended up taking a bit of a gamble for the final submission.
Miraculously, I submitted the one with the best CV score, but I definitely need to reflect on this.

## 3. acknowledgments
I was able to come this far thanks to all the things I learned from everyone.
While I can't mention every single person here, I’d like to express my gratitude to the following individuals.
@siukeitin, @tilii7, @richardjana, @masayakawamata
@cdeotte, @robschieber, @paperxd, @mahoganybuttstrings
@act18l, @gauravduttakiit, @ravi20076, @yunsuxiaozi, @gowthamdd
thanks!!
