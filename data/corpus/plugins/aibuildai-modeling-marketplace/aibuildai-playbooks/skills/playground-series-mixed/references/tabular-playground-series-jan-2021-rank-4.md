# #4 LB Notebooks

Competition: tabular-playground-series-jan-2021
Rank: #4
Source: https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216071

Thx kaggle for hosting.

congrats to @springmanndaniel for 1st place and the size of the score gap between him and everyone else. seen he has already shared solution. look forward to learning from it.

Thx to other participants who shared ideas. I read quite a few public notebooks and I'm looking forward to checking some other solutions.

Did not arrive at anything particularly elegant, but the NN notebooks are shared for interest. Had tried earlier in January without much success. So had more or less stopped on this as it was interesting but I wasn't getting anywhere, then with 4-5 days left read @springmanndaniel in another discussion thread saying his score was based on an NN and figured it was worth a second try. As a result was quite rushed final blend and submission as only got NN CV into a useful range in last couple of days of comp.

**Final submission / score:**
One submission which was the best score I had (CV / public LB). Had no time / submissions for further exploration. Turned out to have best private LB of my submissions.

2 x NN - very similar but the second with fewer features in training, and a closer train/validation loss. The second one added just a small amount to the blended LB score.

NN1 -Private Score 0.69724 Public Score 0.69753
NN2 - Private Score 0.69747 Public Score 0.69809

LGBM, XGB - just used public parameters. These were slightly better than what I'd come up with. I  had some extra features (actually not intentional - copying error) so the scores might not match other notebooks with the same parameters. I don't think these will add anything for anyone as there are a lot of notebooks with better tree-based models & analysis.

LGBM - Private Score 0.69585 Public Score 0.69694
XGB - Private Score 0.69766 Public Score 0.69870

Tried adding Ridge regression with the features I created for the NN. It did a lot better than with original features (down to 0.709 or so), but I couldn't get any positive effect based on my blending calculation.  

Final weights: 

LGBM:    0.374224
XGB:       0.161811
Keras (1)     0.243573
Keras (2)    0.220393

Blended score: Private Score 0.69500 Public Score 0.69579

Probably because I performed some groupings etc across folds, my CV estimate came out a bit lower than LB, but they did both generally keep moving in the same direction. 

**NN Model**
Think this is the only part really of interest though some way off the first place solution and a lot less elegant. Aim was just to get CV low enough to make some useful contribution to tree models.

As noted elsewhere the features look like they have some distributions buried inside them. Used Sklearn gaussian mixture to split original features into sub-distributions (worked better than kmeans). Then for each feature column, split each sub-distribution out into 2 columns: a value (original data) and label (1/0).

Started out with what seemed a 'reasonable' number of splits based on looking at the starting features. However varying it with testing kept showing that more splits improved CV. So end result was a large number of additional feature columns - was not tidy but it seemed to get the data into a format the NN could get better results from.

Did some rescaling of the inputs and reset target around zero - believe this helped though had limited testing time. Network in keras/tf, I started to try pytorch but was having some issues with the results just not looking comparable so given time constraints I left it.

Used optuna at an early point to get something around 5-6 layers with some dropout and dense regularisation, did not have time to update the optuna run while also trying different feature inputs but I think the parameters were probably not too bad as changing a handful of settings on final runs didn't show much improvement. Decreasing the batch size seemed to cause it to learn more steadily but end result didn't seem to look too much different.

As the validation loss bounces around a bit in the last few epochs, saved the weights from last 5 epochs and re-predicted with each of those after finishing training each fold, averaged. This made the CV (on paper at least) lower. Idea borrowed from here (or at least, this is where I personally picked it up from).
https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-inference-tta

For final version ran 4 random seeds, 10 folds. Felt like towards the end more folds or more seeds was diminishing returns on the additional time required so didn't add any more. 

So not very tidy but think it made quite a large difference to LB position by providing a mix from a non-tree model.

**Notebooks**
N.b. I've tidied and re run original notebooks. Scores etc exactly the same.
Thx in advance if anyone has any comments, spots any mistakes I made, obvious stuff that could have been improved etc  

Blend notebook:
https://www.kaggle.com/davidedwards1/jan21-tabular-playground-4-lb-final-blend

NN 1 (more features):
https://www.kaggle.com/davidedwards1/jan21-tabplayground-nn-final-more-features

NN 2 (fewer features, smaller train/valid gap):
https://www.kaggle.com/davidedwards1/jan21-tabplayground-nn-final-fewer-features

LGBM / XGB - not currently added, though the data feed into the blending file is public. Think there are much better kernels for exploring tree models in this comp.

I believe the public parameters I used are from here
https://www.kaggle.com/hamzaghanmi/xgboost-hyperparameter-tuning-using-optuna
https://www.kaggle.com/hamditarek/tabular-playground-series-xgboost-lightgbm
