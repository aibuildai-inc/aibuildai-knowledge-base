# #15 with original data

Competition: playground-series-s3e13
Rank: #15
Source: https://www.kaggle.com/c/playground-series-s3e13/discussion/406395

Hey everyone,

although I am not that close to the top of the private leaderboard, it seems like I am one of the few people who managed to get good results in private (31) and public leaderboard (15) WITH THE SAME SUBMISSION. So I thought a brief description of the tricks I used would be interesting to some people.

The top submissions on the leaderboard probably did not use the original data shown by their big jumps. @zhukovoleksiy obtained great results on the public leaderboard (30.) using original data and also great results on the private leaderboard (5.) with a different model not using the original data.

My models all used the original data but here are some tricks I used to not get biased results:

1. Only use the original data for the fitting of the model and early stopping but not for the model selection. I think this is pretty common.

2. Assign weights to the original data based on their prognosis to counteract the different distribution and artificially recreate the prognosis distribution of the artificial data. To do so the weight for an observation of prognosis p in the original dataset should be set as follows: 
w(p, orig) = (n_obs(p, artificial)/n_obs(p, original) ) * (959/707)

3. Assign higher weights to artificial observations and lower weights to original data by multiplying them with a weight multiplier. I used a hzperparameter for this and tuned it in the grid search. Without this step the original data makes up for 707/959 of the total weights and original data for 303/959. During testing, setting this hyperparameter for original data to 95% and artificial data 5% worked best.

4. Use the weights as sample weights and Eval set sample weights when fitting an xgboost model. You can also use other models but xgboost worked best for me.

I'll try to publish a detailed notebook later.

Thanks for reading, hope this helped some. See you all next week!
