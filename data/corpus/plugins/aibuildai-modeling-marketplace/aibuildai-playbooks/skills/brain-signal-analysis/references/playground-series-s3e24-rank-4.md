# #4 th place solution - robust Hill Climbing

Competition: playground-series-s3e24
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s3e24/discussion/455296

Hi every one, 

I would like to thank Kaggle for this nice competition, and players community, who have shared so many tricks (see below) and so many good models (see below). Congratulation to the winners ! 

And congratulation to @kailai who was at the first place of the public LB during almost all this competition, and who should not have overfit the public LB : only 3 submission. You would have deserved to be the winner !

My strategy was **to avoid overfitting and chance** : public LB was build with only 20% of test data, and this was a 5 digit competition. I wanted to be in the 10% first of the private LB (like I did in the 2 previous PS Competition, but I had not with the same toolkit). 

My solution is based on a trick presented by @oscarm524 at the beginning of the previous competition available [here](https://www.kaggle.com/code/oscarm524/ps-s3-ep23-eda-modeling-submission). He convinced me to try ```Hill Climbing```, **BUT with a robust CV strategy**. I've read @cdeotte about Hill Climbing [here] (https://www.kaggle.com/code/cdeotte/public-lb-1st-place-solution), this was not a 1st place in a private LB but *only* a first place in a public LB.

So I did like I've explained at the beginning of this competition [here](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/450827), by using **Hill Climbing with CV**, with some little improvement (see below).

First I've stored 25 various OOFS predictions, from my own or from public notebooks. With Hill Climbing and CV I've choosen an ensemble of 7 OOFS :
* @paddykb's mean OOFS predictions available [here](https://www.kaggle.com/code/paddykb/pg-s3e24-brute-force-and-ignorance?scriptVersionId=149393795). He didn't use any parameter to fit the ensemble, so there was no risk of overfitting the public LB ; all individual models were very strong.
* @arunklenin's LGB, CAT and NN predictions in version 26 [here](https://www.kaggle.com/code/arunklenin/ps3e24-eda-feature-engineering-ensemble?scriptVersionId=149736443). I didn't take later versions of this notebook because I've seen CV ensemble scores of @arunklenin were not better (maybe I'm wrong). I trained all fonds with all original samples, I had OOFS output for every model (and preds on test too), because the oof prediction available in original output had been build without CV. I've used oofs predictions of each model instead of using the optuma oof prediction.
* I had a personal LGB and 2 personal XGB (no feature engineeing, origin dataset added for training, ```hearing(left)```, ```hearing(right)```, ```urine protein``` dropped, random grid search, with GPU for grid search of XGB, grid search with 4 folds and repeatedstratifiedkfold for the 20 best models)

I've trained Hill Climbing with 5 folds and by taking care that : 
* when a model improved AUC score on a train fold, there is a score improvement on a validation set too. 
* that the 7 models improves score in all validation folds
* and i ve repeated this experiment with 5 other folds (another seed) to be sure.
The 3 first models (with the strongest coef) in Hill Climbing where always in the same order : @paddykb first, then @arunklenin's LGB and NN

**About Hill Climbing improvements** during this competition : 
After I had added parallel computing during Hill Climbiing and CV, @siukeitin had helped me to optimize Hill Climbing AUC computation with numpy [here](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/450827). 

See you later and have fun !
