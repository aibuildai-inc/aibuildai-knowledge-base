# 6th place solution

Competition: hungry-geese
Rank: #6
Source: https://www.kaggle.com/c/hungry-geese/discussion/263916

Thank you for hosting such a fun competition. 
For this competition, I used the awesome [HandyRL Library.](https://github.com/DeNA/HandyRL) 
It was easier to read and study the code than other libraries. 



### 1. Input
I used 3 inputs as below.
[inputs]

### 2. Model

[model]

### 3. Ensemble and Post Processing
I did argmax in the sum of the output of the LB top 3 models.
If the Action is in the opposite direction of the previous Action, the argmax is taken excluding that Action.


### 4. Validiation and submission
For each check point, 100 games are played with 1 imitation learning and 2 Public Agent “Smart Geese Trained by Reinforcement Learning”.
Scores were given 1st place +5, 2nd place +1, 3rd place -1, 4th place -5 points.
Among the new checkpoints every day, the one with the highest sum of validation scores was submitted.
