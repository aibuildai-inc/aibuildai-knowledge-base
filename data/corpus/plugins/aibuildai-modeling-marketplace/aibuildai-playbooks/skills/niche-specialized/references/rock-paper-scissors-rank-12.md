# 12th place solution - lightGBM model

Competition: rock-paper-scissors
Rank: #12
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221519

Fortunately, I achieved 12th place in this competition.  
This is a very interesting and fun experience for me.  

I would like to share my solution in this competition.

This is my best agent:  
[https://www.kaggle.com/watanabekazuo/12th-place-solution-lightgbm-model](https://www.kaggle.com/watanabekazuo/12th-place-solution-lightgbm-model)

## Base Strategy
I use 4 lightGBM predictors training from last 50 matches.

- predict next (my / opponent) move  
- input is last (2 / 3) matches

And there are 3 strategies for each predictors, using win hand, draw hand and lose hand.

Therefore, there are 12 different strategies.  

After each match, change the weight of each strategy.  
If agent win the last match, the weight of strategy of last match will be increased, and if lose, weight will be decreased.  
And change the weight of related strategy (which use same predictor)

## Additional Strategy

### Confirm Method
If the predictor outputs very similar probability (e.g. [0.30, 0.35, 0.35]), is this result useful for evaluate strategy?

So, I set "confirm threshold".  
If the highest probability > threshold, agent will use this strategy. Otherwise, agent use random strategy.

### Safety Method
For example,  the predictor outputs opponent move probability [R: 0.45, P: 0.50, S: 0.05].  
In this situation, "S" is best beating strategy, but "S" will lose in 45%.  
And "P" is not best beating strategy, but "P" will win in 45%, draw in 50% and lose in 5%, this is a reasonable strategy.  

Therefore, I set "safety threshold".  
If the lowest probability < threshold, agent will use this strategy.


## Hyperparameters
There are many hyperparameters.

For example:
- Confirm method threshold
- Safety method threshold
- Change rate of strategy weight
- Number of match used for training
- lightGBM hyper parameters

Everyday, I little tweaked hyperparameters, and submitted agents.
