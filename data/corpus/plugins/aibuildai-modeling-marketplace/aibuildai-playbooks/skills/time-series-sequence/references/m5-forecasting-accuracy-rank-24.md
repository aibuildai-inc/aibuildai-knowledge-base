# A few thoughts - 24th Place

Competition: m5-forecasting-accuracy
Rank: #24
Source: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163208

Like a few others I was one of those that benefited from the leaderboard shakeup. For reference I didn’t use any multipliers. My solution I think was relatively simply, being a combination of light gbm and a simple LSTM. The biggest benefit I found was in getting local WRMSSE working which allowed me to do a better job tuning and also where I found the biggest benefit in feature selection especially given the large number of features I’d engineered throughout the competition. 

I think one of the challenges with this competition was the evaluation metric. While I did work with a custom objective function during training a lot of different approaches I tried often didn’t have the outcomes I expected when evaluating. 

In the end I found my cross validation to be fairly consistent by the end and was close to the score I got on private LB. Incidentally my model would still perform poorly on the public leaderboard against some scores on there.

It would also be interesting to understand if the hosts actually picked this particular  period for scoring on purpose or they were just arbitrary. Was there some nuances in this period that other forecasting methods have struggled with and that’s why they were selected?
