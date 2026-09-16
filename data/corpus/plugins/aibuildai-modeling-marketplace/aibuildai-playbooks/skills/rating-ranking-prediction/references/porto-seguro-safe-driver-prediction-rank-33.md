# 35th place solution

Competition: porto-seguro-safe-driver-prediction
Rank: #33
Source: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44711

Here is my solution for a 35th position. Nothing special here just a blend of different models.

**Cross-validation**

I used StratifiedKFold with seed 15 and saw early on that fold 2 and 3 were closely linked to Public LB. Some models were closer to Fold 2 and others were close to the average of Fold 2 and 3.
I could increase my LB position while checking my overall local CV improvement and hope for the best on private LB.

**Feature engineering**

2 and 3 way feature interactions tested with a simple SGD.  Continuous data was binned using pd.cut (did not have time to test Tilii's binning strategy but will in the future)
All was then One Hot encoded to sparse matrix, removing small occurences ( &lt; 100 ).
My best SGD scores 0.28206 on private LB.

I tried to use a simple distance to mean of positive samples and mean of negative samples. The distance in itself was able to score 0.23828 on public LB.  But for some reason models were unable to build on this... 

The only thing I found in this dataset is this:  suppose you train an xgboost on the full train set and you get 0.286 local. If you check the scores for samples where ps_car_03_cat is equal/different to -1 you have a substantial score difference like 0.23 or 0.24 for one part and 0.30+ for the other samples. This is also true for ps_car_05_cat and ps_reg_03
Training models on these parts of the dataset did not really give local improvements...

**Models** 

- LightGBM with 300+ features (OHE, target encoding, frequency, you name it) Local CV 0.2863 and private LB of 0.2869
- XGBoost with less features. It seems LightGBM sklearn API is better at managing memory. Local CV 0.2865 and private LB 0.2875
- Regularized greedy forest (XGBoost features). Local CV 0.282 and Private LB 0.284
- LGBM in Random Forest mode. Local CV of 0.2738 and Private LB 0.2732
- Keras 2 layers, I'm not very good with NN as I find them too long to train (I don't have a Tilii Special Nvidia GPU ;-), you know the TI ones) This was using OHE sparse data and 5 bags on each fold. 1st layer 50 PReLUs and 2nd layer 25 PReLUs,  L2 regularization, No dropout. Local CV 0.277 and private LB 0.282
- LibFFM - a big thank you to Scirpus and Tilii. The only thing I did here is generate the files for each fold and simply run the exe file. Local 0.2816 and private score 0.28597
- FTRL proximal - another big thank you to Scirpus and Tilii. Local CV in the 0.269 and private LB 0.273
- SGD and Ridge, both on One Hot Encoded data. For ridge I used a sigmoid to transform the decision_function. 

**Ensembling**

I used a linear stacker of my own that swaps meta features to maximize/minimize a given metric (and no intercept). And yes weights can be negative ;-) In my experience allowing negative weights always give the best results. The submission was an average over all folds with a rescaling procedure that uses a LogisticRegression.
It scored 0.29085 on local CV and 0.29113 on private LB.

XGBoost gave substantially worse local CV but scored 0.29083 on private LB so very close to linear stacking.
