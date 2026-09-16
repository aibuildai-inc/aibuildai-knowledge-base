# 4th place solution - stacking approach with XGB as meta model.

Competition: playground-series-s3e26
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s3e26/discussion/464863

Thanks for the PS3 2023 series, always a fun and good area for testing and experiment new models and ideas, benchmark against the old ones! It started well and ending the same with a 1st in PS3E1 and 4th place in this one, the common for all during the series are the knowledge one gets, valuable knowledge and insights.

**The solution**

As hinted in this post https://www.kaggle.com/competitions/playground-series-s3e26/discussion/461062 XGB works well in classification, and that was the base idea.
The key was using a stacking approach with OOF from many other different top solutions and use XGB as meta model for training the stacked predictions + orginal features. And also try not to tune and optimize every solution to much, reduce the risk of overfitting, instead use the predictions as extra features in the final stacking training.

Many solutions shared the same small added feature engineering as below, rest was unchanged in terms of FE.

train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[9000, 15000, 20000, 25000, 30000], labels=['A', 'B', 'C', 'D'],)
train_df['Log_Age'] = np.log1p(train_df['Age'])
scaler = MinMaxScaler()
train_df['Scaled_Age'] = scaler.fit_transform(train_df['Age'].values.reshape(-1, 1))

**The different solutions and frameworks trained**

**AutoGluon**

AutoGluon latest pre-release 1.0.1b20231208 with zero-shot HPO as default. The trained framework used below weighted ensemble of models. I also tried distillation and pseudo labeling but didn’t work better. 

0	WeightedEnsemble_L2	-0.436142	log_loss	7.002513	505.384119	0.004281	9.923933	2	True	9
1	XGBoost_r89_BAG_L1	-0.441355	log_loss	0.470275	19.330144	0.470275	19.330144	1	True	6
2	CatBoost_r137_BAG_L1	-0.441618	log_loss	0.138598	115.856482	0.138598	115.856482	1	True	4
3	CatBoost_r50_BAG_L1	-0.442764	log_loss	0.266119	78.536804	0.266119	78.536804	1	True	8
4	LightGBM_r130_BAG_L1	-0.443168	log_loss	1.180046	49.676369	1.180046	49.676369	1	True	7
5	XGBoost_r33_BAG_L1	-0.451191	log_loss	3.457726	68.160608	3.457726	68.160608	1	True	3
6	RandomForestEntr_BAG_L1	-0.475263	log_loss	0.428620	3.049245	0.428620	3.049245	1	True	1
7	NeuralNetTorch_r79_BAG_L1	-0.482747	log_loss	0.301830	66.300480	0.301830	66.300480	1	True	2
8	NeuralNetFastAI_r145_BAG_L1	-0.489281	log_loss	0.755018	94.550054	0.755018	94.550054	1	True	5

**LightAutoML**

Trained the LightAutoML 0.3.8b1 version and the trained frameworked used below weighted model ensemble.

[16:19:15] Model description:
Final prediction for new objects (level 0) = 
	 0.06558 * (5 averaged models Lvl_0_Pipe_0_Mod_0_LightGBM) +
	 0.17057 * (5 averaged models Lvl_0_Pipe_0_Mod_1_Tuned_LightGBM) +
	 0.27900 * (5 averaged models Lvl_0_Pipe_0_Mod_2_CatBoost) +
	 0.48485 * (5 averaged models Lvl_0_Pipe_0_Mod_3_Tuned_CatBoost) 

[16:19:15] ==================================================
[16:19:15] Blending: optimization starts with equal weights and score -0.4164338072355177

**AutoXGB**

Trained AutoXGB 5 fold CV with standard settings but with the extra features.

**Public notebook**

https://www.kaggle.com/code/mattop/ps-s3-e26-lgbm-xgb-preprocessing-fasteda
Re-trained the notebook and saved the OOF for the XGB and LGBM. Noticed a good local CV that could be good to use in the stacking and also as diverse model for the owned created.

**Other trained solutions**

Trained several other SOTA solutions and ideas, like using old top solutions from old competitions multiclass and binary as well, train every class separately and use the predictions as extra features. It worked well but the four solutions below was enough looking in the mirror.

**Stacking**

I then trained the final 20 fold XGB model as meta model with the original features, the extra created and OOF predictions, both the CV score and the leaderboard went better. I also trained a 2 level stacking approach with several of tested stacked XGB models, as an approach instead of ensemble for the second submission.

--------------
**That’s it! Happy Kaggeling!**
