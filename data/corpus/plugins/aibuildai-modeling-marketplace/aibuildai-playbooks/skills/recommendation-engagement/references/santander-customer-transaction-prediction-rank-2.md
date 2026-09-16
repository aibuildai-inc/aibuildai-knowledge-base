# 2nd place solution

Competition: santander-customer-transaction-prediction
Rank: #2
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88939#latest-528874

First, I don't want to write this title. I'm fed up with 2nd place.

But I would share our solution briefly.

1. remove fake from test
2. concat train and test, and then invert some of features
3. standard scaling
4. count encoding
5. count `round` encoding
6. unpivot all vars(so we have 200k x 200 = 4m train samples)
7. train and predict
8. convert prediction(200k x 200) into odds. We used `(9 * p / (1 - p))`
9. submit
10. press 'F' to pay respects

I will share codes later. Thanks,

====================================
Edit1: our best model is `NN: LGB 3:1`

Edit2: 
my github: https://github.com/KazukiOnodera/santander-customer-transaction-prediction
Golf: https://github.com/KazukiOnodera/santander-customer-transaction-prediction/blob/master/py/990_2nd_place_solution_golf.py

Edit3: confirmed code golf score
[pic]
