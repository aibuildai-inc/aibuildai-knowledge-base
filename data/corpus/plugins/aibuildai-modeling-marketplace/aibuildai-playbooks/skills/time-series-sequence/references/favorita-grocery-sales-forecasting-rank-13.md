# 13th place solution

Competition: favorita-grocery-sales-forecasting
Rank: #13
Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47542

## Feature Engineering

After reading a couple of solutions posted, I think this is my biggest trump card compared to the others. I started by thinking about my own grocery shopping habit.

* I tends to shop on a weekend. (timing)
* I will stock up if things I want is on promotion. (on promotion)
* If I already have enough of something at home, it's much less likely for me to buy more even if it is on discount. (historical sales)
* If I have enough beef at home, it's much less likely for me to buy more any other type of meat as well (sales by class)
* If beef is on discount, then I am less likely to buy chicken (promotion by class)

The key insight is that most people maintain a certain level of stock for a particular group of goods at home, not individual goods. I think most people missed this. Or in classic economic theory, there is a substitution effect among goods. I believe including these class-based features boosted my model by a large margin. I added these features at the very early stage so I don't have a good estimate on how much the impact this.  

My formal education is in economics which definitely helped, but at the same time, I think these features are pretty generic with limited domain expertise involved. 

## Model

I used sliding window approach like everyone else. 

My base model is a simple DNN. started with a batch norm layer, then followed by 6 dense layers with a .3 dropout layer between every two. Activation with 'elu' with the only exception of last layer of 'relu' to force the output to be positive.

I started with train the model using the last 2 years data and after a lot of tuning the best public score, I managed to achieve with this 0.510. 

I stuck at 0.510 for a long time then because I have been training with RMSLE without weighting, and my intuition also tells me, the behaviour for perishable and non-perishable might be different, I decided to train the perishable and non-perishable separately. Doing this, I managed to improve the public score by another .001.

My final model is an ensemble of the above two variants and AhmetErdem's public kernel. this improved the public score to 0.507


## Validation

Initially, I used the last 16 days across all series in train dataset. And the performance has been quite consistent compared to the public score.

However, after I started training the perishable and non-perishable, I noticed further improving my local cv actually decreased my public score so I added another 16 days to my validation set, which turns out to be quite helpful in training the perishable goods.
