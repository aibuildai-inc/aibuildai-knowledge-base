# 15th place solution

Competition: ieee-fraud-detection
Rank: #15
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111454

As a first thing, I want to thank both Kaggle and IEEE for letting us to compete in this challenge. It was a quite exhausting but also educative process for me.

I've already read all of the top solutions and I know that I don't have super different tricks from theirs. I just wanted to tell how my process shaped during the competition for those who are wondering.

Let's start.

My final submission has following main points:
- Generated my predictions by using unshuffled KFold. Since the test set was time split and I was using unshuffled KFold, I mimicked all these for my local validation scheme. I separated last 20% as hold-out and got predictions for it by setting another unshuffled KFold with the first 80% of the train data. Tested all operations like feature generation, parameter tuning, post processing etc. with this strategy.
- Only used LightGBM. I was neither able to generate sufficiently good CatBoost predictions nor NN predictions. Final sub is 4-5 different lightgbms' weighted blend.
- I couldn't perform any feature selection because all the methods I tried worsened my CV.
- Found different userid's based on different feature combinations. And tested their quality by calculating target means after shifting isFraud by 1 within train data.
 - Example: 
```
train['ReferenceDate'] = train.DayNumber - train.D1

key_cols = ['TransactionAmt','ProductCD','card1','card2','card3', 'card4','card5','card6','addr1','addr2','ReferenceDate','P_emaildomain']

train['key'] = train[key_cols].apply(tuple,1)

train['previous_isfraud'] =  train.groupby(['key'])['isFraud'].shift(1)

print(train.groupby('previous_isfraud')['isFraud'].mean())
&gt;previous_isfraud
&gt;0.0    0.000528
&gt;1.0    0.991961
```

- Shifting isFraud info from train to test rows for the same userids and applying post process was the main boost of course. Attention: Postprocess was not directly shifting isFraud info from train to test. It was multiplying predictions with a constant number which I found by using validation dataset. Since I had multiple different userids, I also had multiple postprocessing steps.
- Creating many aggregated features based on userids.
- Assigning mean of predictions to a userid's all transactions. This gives significant boost indeed. (Applied this only for userids with high accuracy)
- For blending, I found that lgbm boosted with 'dart' method was quite useful.
- Applied final blending on **only new users** in the test. 

**What were my milestones?**
- Analyzed dataset on Excel by sorting TransactionAmt and card infos. Seeing the high correlation of the shifted isFraud with real isFraud  for the sorted dataset. (I find it quite useful to look at data in Excel most of the time.) 

- Realizing the real meaning of D1 and D3. Then creating 'ReferenceDate' feature by simply calculating data['DayNumber'] - data['D1']

- Starting creating userid's by using different feature combinations and using sihfted isFraud info for postprocessing of final predictions.

- Adding testrows with shifted isFraud info coming from userids with highest accuracy as psudeo datasets during training.

- Creating aggregation features based on userids.

- Realizing that our models were learning by overfitig actually. Isn't it weird that numleaves &gt;= 200 was performing quite well ? It was because the models were actually finding same users in the test set. So we had real problem at new users which only exist in the test set. You can also check this for your models. Take last 20% of the train data as hold-out and make predictions. If it has around 0.940 AUC, now also check AUC for users also exist in train and for users only exist in hold-out. You'll see that old users have AUC around 0.977 and new users have AUC around 0.900.

- Last week I also focused on improving new users in the test set, because private LB has much more new users than public LB. Found out that some of the features that I generated was improving old users' predictions but lowering new users' predictions. Also for the same model, increasing regularization and lowering num_leaves led to better new user predictions. So I rerun my main model for new users with dropped features and new model parameters. Also moved from unshuffled KFold to GroupKfold grouped by userid.
 
- Finally used new model predictions to blend my best submission but only for new users. Did not touch old users' predictions. I think this process gave me around 0.001 boost. I wish I had worked on modeling by considering two types of users from the very beginning.


P.S.
As a solo competitor, being silent for a long time to have competitive advantage and to avoid private sharing was hard indeed. I like to talk when I found something immediately :d.
Also I know that the forum and kernels are actually quite beneficial, however; when you found something before others you just pray everyday not to see another post saying 'there is something in V columns.. How to find true users..' etc. etc. 😆 😆 

Have a nice weekend and
See you in the next competition!
