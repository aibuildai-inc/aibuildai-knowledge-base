# 9th place solution notes

Competition: ieee-fraud-detection
Rank: #9
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111234

I know, 9th place out of 6000+ is a very good place. But well... it doesn't exactly feel so after being 1st on public LB for quite a long time :)
Anyway, this was an interesting competition and - it's better to fly high and fall down, rather than never try flying.

 

First I would like to congratulate top teams and especially **FraudSquad** for well deserved 1st place with big gap from others!
Also special congratulations to some people I know better than others - my ex teammates from other competitions @johnpateha and @yryrgogo for getting in gold!

 

And of course, the biggest thanks to my teammate @kostoglot for great and professional teamwork!

 

##Some key points from our solution

* We heavily used identification of transactions belonging to the same user (as I think all of the top teams did).
* To identify users very helpful feature was ("2017-11-30" + TransactionDT - D1) - this corresponds to some date (like first transaction date, etc) of card - same for all transactions of one card/user. Similar applies to several other D features.
* To check identified users very helpful features were V95, V97, V96 which are previous transaction counts in day, week, month and features V126, V128, V127 - cumulative amount of transactions made by user in previous day, week, month.
* We have different approaches on how to integrate user identification in solution - Konstantin used them as features, I used them in postprocessing and pseudo labelling.
* We used cv setup taking first training data set 3 months for training, 1 month gap removed and 2 last months for validation. This cv setup correlated quite good with public LB, but as it turns out, in some cases it lied to us regarding private set and at some points we moved in a bit wrong direction resulting in falling down to 9th place.
* Models used were LightGBM, Catboost and XGBoost.

Good luck!
