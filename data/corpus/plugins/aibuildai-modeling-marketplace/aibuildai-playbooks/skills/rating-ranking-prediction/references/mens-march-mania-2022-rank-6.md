# 6th place solution, team embedding

Competition: mens-march-mania-2022
Rank: #6
Source: https://www.kaggle.com/c/mens-march-mania-2022/discussion/317114

Hi everyone, here is my work: [Notebook](https://www.kaggle.com/code/axotcc/6th-place-solution-team-embedding?scriptVersionId=92171964)

Congrats to me to get the first gold medal, it's so close to the prize though🥺. However, my method only got 472th/659 in [March Machine Learning Mania 2022 - Women's](https://www.kaggle.com/competitions/womens-march-mania-2022).

In my own knowledge, my method is different from the previous works that most of them are doing feature engineering to add and aggregate features as input and target for common machine learning models. In my work, I try to train team embedding of the given team_id and make model to predict the competition result from given two team embeddings.

The following express my method which is copy from my notebook.

### Model

[drawing]

- The goal of model:
    - Given 2 team_id, predict the competition result
- Input:
    - **team_emb**: trained embedding, represent the embedding of given team_id, initialized with torch.nn.Embedding().
- Output:
    - **team1_win**: whether team1 win the game
    - **team_info**: abbreviation of the competition detail result, contains ['Score','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF'] 


### Data
- **Training data**: 
    - RegularSeasonDetailedResults.csv
- **Validation data**: 
    - NCAATourneyDetailedResults.csv

### Preprocess
1. Duplicate the data and make it symetrical to get rid of winner and loser.
2. **Add feature** "**is_win**" which represent whether team1 win the game.
2.  Apply **quantile transformation** to transform the **team_info** to normal distribution.

### Training
1. Train for one epoch along the timeline
2. A minibatch is the data that has the same 'Season' and 'DayNum'.
3. The team embeddings will be updated everyday.


### Future work
- Apply GNN to let a single game result of two teams affect other teams and enhance the ability of the model to apply the relationships between the teams.
