# [7th Solution] Ensemble of Tree + NN

Competition: um-game-playing-strength-of-mcts-variants
Rank: #7
Source: https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549617

CV 
Well since for me sometimes GKF with Game more algin on cv LB and sometimes GKF with GamerulesetName better. I am a bit confused and in the final days I stick with GKF + Game + statify with label, for I think for test set **new Game dominate**, but this might be a wrong decsion as PB show my final ensemble weights using this CV not work at all.
- Data aug
Like other teams I have also used flip aug (swich agent1 agent2, advantageP1,label) at very first and it show improve on LB about 0.01
I also used test time TTA by combining flipped data.
- Tree models  
For me tree model work better then nn and tree model results is much more stable on LB. 
The key for tree models to perform well is using 
1. **manual cross feats  of numerical feats** like * / of top 20 numercial feats  
  This improve both cv and LB a lot, it keep improve using top 5 , 10 15 until 20, then more cross like top 25 * 25 will hurt LB.

2. Target encode help LB 0.001
3.  Make tree deeper for lgb I used 16 and for cbt I used 10.  
4. To avoid overfit set reg lambda high like 10, set random strength of cbt high like 10.  
5.  Dart mode for lgb improve cv but hurt LB, my cv GKF + Game might not correct here.  Well trust LB...
6. Tree results:  
  LGB single model 10 folds LB 419  PB 425 
  CBT single model 5 folds   LB 422  PB 428 I tried so much effort in the last week to improve CBT model however it not contribute to final PB.
  Well simple ensemble of LGB + CBT improve LB only a bit less then 0.001 but not improve PB.
-  NN models
Follow the https://www.kaggle.com/code/yekenot/mcts-deeptables-nn this great notebook.
I used torch to impement the NN model with top numeric feats turn to bins and adding lgb tree leafs feats(depth 5, 200 trees).
Also used CIN + FM for the pooling layer and used label aug(mutinomal) for nn model.
However nn model not stable for LB and work worse on CV (GKF + Game)
By luck I have one 5 folds NN model with LB 419 PB 427, notice due to randomness nn model mostly get LB about 423, but since LB and PB align you just use your best LB nn model.
- Post deal  
Optimize oof a * x + b for overall rmse 
-  Ensemble  
My best LB 416 PB 423 model is simple blend of LGB(10 folds) + NN (5 folds) with weights 1, 0.75.

If I have more time, I will try to swich back to cv with GKF  + Gamerulesetname and test CV + LB and try stacking with OOF method.
