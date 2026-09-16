# #38 solution

Competition: santander-customer-transaction-prediction
Rank: #37
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88967#latest-513833

祝贺各位老哥!
Congratulations to all winners
thanks all my best teammates @chizhu @puckW @zhouqingsongct 
here is our simple solution
solution ordered by time 
# 904 
just freq encode all vars by train ,then for test part all freq var add 1
u can achieve #904 in lb
catboost model
# 914
just remove test fake
then use train + test  to get freq encode 
# 922
just original features and original features replaced by median when freq var ==1
# 923 and 924
just augment and stacking

what a pity that we forget add weight of our nn to our stacking/blending models

finally
Thank organizations
thanks niubility share in discussion

感谢机构感谢各位选手牛逼的分享
u can translate Chinese to English because my English is poor
thank anyway
we will post our code later
this kernel might be overfitting a little ,but it show our core method for this compe
https://www.kaggle.com/chizhu2018/augment-catv2?scriptVersionId=12597845
