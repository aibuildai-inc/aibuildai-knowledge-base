# 13th place and how to

Competition: open-problems-multimodal
Rank: #12
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/366504

Hey guys, it's been a while since my last actively participated competition and this is a good experience! Though I am not active any more, I frequently come back to kaggle to browse new ideas. One thing I notice is that it is not really straightforward to learn new things by just reading top solutions if you were not active in that competition... And one factor could be that we mostly share what finally worked (and/or what didn't work), but not the thought process of getting there. For someone like me, either a bit too lazy or a bit too busy to actively participate, or someone who might be a bit inexperienced, I think it would be more beneficial to share "how I got here" more than "here I am". So in that spirit, I would like to start sharing solutions this way...


##### Some background: 
My job can be demanding so whenever I can delegate the work to computers, I do, whenever I cannot, I minimize the time needed for coding such that I can leverage fragamented time as much as possible. And I mainly participate in this competition to learn how feature extraction could work with high dimension inputs and outputs. And they shape what next experiments I decided to try, and mistakes I made along the way so I thought it is important to share.


##### My journey:

1. started by reading the Discussion to understand what the data is about, and walk through popular public kernels to understand what can be used as baselines. I noticed that mostly TruncatedSVD was used to reduce dimensions. 

2. I figured it might be a good idea to just train a mlp model with all features included. And I was too lazy to build a local validation pipeline so I just randomly sampled 10% as validation set. And the result wasn't so good on public LB compared to public kernels.

3. Then I was thinking, ok, maybe it was because mlp was bad. But xgboost/lightgbm on cpu would take forever to train, and would run out of memory on gpu. So need a better neural network model. What about Tabnet (terrible). Ok. There was one paper I remember claiming similar performance to xgboost. OK. found it. https://arxiv.org/pdf/2112.02962.pdf (It is called DANets). Better but still not as good...

4. Now back to public kernel as the baseline. Maybe instead of SVD, we can use autoencoder? OK, only linear autoencoder  performed ok-ish, any nonlinearity didn't work... no matter what tricks (e.g. swap noise augmentation) used. Hmmm...

5. Back to TSVD + MLP as baseline again... Let's make this baseline better first. First swap MLP with DANets. And let's just focus on cite since it only have high dim inputs. Whatever works for cite should work for multi right? (TimeMachine: Nope!)

6. Let's standardize the data since PCA likes it. OK. Slightly better. The explained variance seems quite low, but adding more components as features doesn't seem too helpful. So likely the inputs are quite noisy. Don't really know what to do. Well, we can always add different decomposition methods if no better ideas. Added NMF, FactorAnalysis, FastICA. Ok. All of them worked. 

7. Now let's also train some xgboost model since now we can train on gpu. Ok cool. Averaging xgboost with DANets improves results significantly.

8. Maybe should try autoencoder again.... Read some papers. Ok. Still didn't work.

9. Let's try some popular nonlinear dimension reduction techniques. UMAP, TriMAP, PaCMAP. Doesn't seem working.

10. XGBoost or LightGBM train one model per target which seems wasteful and may not consider the correlation between outputs. Let's see if there is a better way out there. https://arxiv.org/abs/1909.04373 found this GBDT-MO. and it has code. Tried. Didn't work so well. 

11. Read in the Discussion that we can select features by matching input/output names for Cite. Tried and it worked. Neat.

12. Also read in the Discussion that the 0s in inputs may not be actual 0s could also be missing. OK. Tried to calculate the mean/std by considering all 0s as missing and then calculate PCA. Adding to the features improved the model a bit.

13. Realized that if 0s can be treated as missing then we can calculate PCA differently too according to this old paper (https://www.sciencedirect.com/science/article/abs/pii/S016974399600007X). Helped a bit.

14. OK... stop the laziness and spend the weekend on building proper cross validation pipeline and retrain the models for cite. Nice. a huge jump from 0.812 to 0.814 on leaderboard.

15. Saw some discussion about the supplymentary raw data. Well, I believe the host has done the best preprocessing and probably not helpful, also it is a pain to handle two datasets, so ignored. (TimeMachine: Big Mistake!)

16. Approaching the last week of competition so rewire the cite pipeline for multi. Hoping to see a huge jump on score for multi as seen on cite. But that didn't happen. So... They are actually very different... Maybe I should have accepted some team merging invite earlier....

17. stack a few DANets and GBDT models. 

18. picked the wrong submissions for final evaluation. But what can you do.


##### Model Summary
Cite:
(PCA + NMF + FA + ICA + NanPCA + Missing Value PCA (i.e. NIPALS) ) + (DANets + XGBoost)

Multi:
(PCA + NMF) + (DANets + XGBoost with SVDed Output)


##### Looking Back, to improve the score further
1. Should have spent more time on data preprocessing.
2. Should anticipate and prepare for a complex ensemble pipeline so save all the models and predictions properly along the way for multi layer stacking.
3. Shouldn't have assumed Cite and Multi being similar and go for team up.


Hopefully it is helpful (also to those who didn't participate!)
