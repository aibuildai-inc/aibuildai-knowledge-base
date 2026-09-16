# 25th Solution

Competition: ieee-fraud-detection
Rank: #25
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111511

Thank you for Kaggle, Vesta, IEEE-CIS for hosting this competition. And before writing about the summary of our team's approach, I really want to appreciate my team-mates,( @yeonmin, @chocozzz, @soonhwankwon , @sunghun), without your hard work it wasn't possible to end up well. All of my team-members needed a gold medal to become a competition master. That's why we choose "borabori" (Tinky Winky in Teletubby) as our team-name, but we missed it by 2 positions. Let's promise it next time :D
  
# Overview
Our team made 2 independent pipelines. Two different approaches were quite different, so these gave us huge boost when ensembling(stacking wouldn't work for us). I'm going to write down the main concepts of each pipeline below.

# First model
-  Specify the Client_id :  At the very first stage, we found out the real meaning of D1 column, so we could specify client_id by (card1, addr1, addr2, make_account_date_D1), though it was not perfectly matches. 
- Again Client_id, Card_id : After roughly finding the combination for discerning client, we thought it would great if we figure out how to specify the following two groups. (1) Who is the owner of each transaction(=client_id), (2)Who is the owner of the card that was used for each transaction(=card_id). Users can have more than one card. If specifying these two is possible,  we thought grouping the transactions by card_groups or client_groups, then making aggregation features from them would be the best way to solve this problem.(By implementing GroupKfold based on card_groups or client_groups)
- Failure : By the last day the competition ended, we couldn't find the perfect condition for specifying groups. So, we generated diverse client_id, card_id combinations to approximate the real client_id, card_id.
- FE : Created aggregated, lag, lead features based on those "imitated combination" such as how many unique items(=ProductCD +'_'+TransactionAmt) are pursed by each client, how much was the time gap between the previous transaction and the current one. And frequency-encoding for some categorical variables.
- Feature Selection : First of all, we only took the common values between Train/Test-set for categorical variables, and replace the uncommon values with NaN. Then, removed some columns by adversarial validation check.
- Validation Strategy : As I mentioned above, we would have gone with GroupKfold based on client_groups or card_groups. But we couldn't. Instead, we choose validation strategy as  GroupKfold(K=6) based on the month_block. 
- Model : LGBM(goss, dart, gbdt) , CATBOOST, XGBOOST. For the model performance, (CAT&gt; LGBM &gt; XGB). The single model's score was varied a lot depend on seed. So, we trained each model at least with 5 different seeds.

# Second Model
- Specify the Client_id ,Card_id 
- LDA, NMF : First change the data-type as string and concatenate, then implemented LDA for two combinations( ['card_1','addr_1'], ['card1','id_20']), NMF for one combination(['card4','DeviceInfo']).
- FE : Similar job with the first approach + Target_mean_encoding for some categorical variables
- Validation Strategy : First, split the dataset into two sub-groups. One for "ProductCD" is "W", and the other for "ProductCD" is not "W". Then, implemented StratifiedKFold(K=5) for each sub-group. 
- Model :  LGBM(goss, dart, gbdt) , CATBOOST, XGBOOST. For the score, (CATBOOST &gt; LGBM &gt; XGB). 


# The things didn't work for us.
- NN : @soonhwankwon really tried hard for this part. But it was hard to make a robust, good-scoring model with NN. And also, ensembling/stacking the result with tree-based boosting model didn't work.
- DAE 
- Stacking
