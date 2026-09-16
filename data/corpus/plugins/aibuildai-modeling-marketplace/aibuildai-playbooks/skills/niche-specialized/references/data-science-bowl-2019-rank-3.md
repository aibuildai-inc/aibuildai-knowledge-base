# 3rd solution - single TRANSFORMER model, link to kernel

Competition: data-science-bowl-2019
Rank: #3
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127891

I'm sorry. It was a traditional Korean holidays until today, so I didn't have time to write this. Thank you for your patience.

First of all, I would like to thank Booz Allen Hamilton for hosting this interesting competition. And congratulates to the all participants and especially the winners! 

I like deep neural networks so I prefer to solve all the problems with a deep neural networks. 😃 

I focus on the structure of the input data rather than understanding the input data. And concentrate on making the model's input by avoiding missing information as much as possible, hoping that the model will do more than I expected. 😊 

In other words, I focus `less` on feature engineering and `more` on finding a neural net model architecture that fits the data.


# Interesting point
- What's interesting is that using position-related information(especially position embedding) decreases local CV score.
	- The performance of the BERT, ALBERT and GPT2 models was not good. (Because these models use position embedding)
	- So I used the TRANSFORMER model without position embedding.

# Pre-processing

### Aggregation by game_session

The sequence of installation\_id is too long to be used as it is. So I aggregated log data (train\_df) by game_session. Please see the example below.
```
df = train_df
event_code = pd.crosstab(df['game_session'], df['event_code'])
event_id = pd.crosstab(df['game_session'], df['event_id'])
...
agged_df = pd.concat([event_code, event_id, game_accuracy, max_round])
session_df = df.drop_duplicates('game_session', keep='last').reset_index(drop=True)
session_df = session_df.merge(agged_df, how='left', on='game_session')
```

The LSTM and TRANSFORMER models in NLP receive sequence of words (or sentence) as input. Similarly, I will use the sequence of game\_sessions (or installation\_id) as input here. 


# Model

Best private score: 0.564
Single transformer model used.

### TRANSFORMER MODEL BLOCK



Prediction from game\_sessions of an installation\_id

### The key here is how to create embedding from the game_session.

`Categorical columns` (such as event\_code, title, world, etc...) were embedded respectively. Then, the categorical_vector were obtained by concatenating the embeddings. Next the nn.linear layer is applied for the dimension reduction of the categorical vector.
```
self.categorical_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*num_categorical_columns, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )  
```

`Continuous columns` were embedded directly using a linear layer.
```
self.continuous_emb = nn.Sequential(                
            nn.Linear(num_continuous_columns, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
```
* I used np.log1p for normalization of continuous columns.

### hyper parameters
- optimizer: AdamW
- schedular: WarmupLinearSchedule
- learning_rate: 1e-04
- dropout: 0.2
- number of layers : 2
- embedding_size: 100
- hidden_size: 500




# Modified loss function

https://www.kaggle.com/c/data-science-bowl-2019/discussion/124836
As mentioned in this link, the 0 and 3 classes of the accuracy_group may be very close. 
num\_correct can have 0 or 1, if the num\_correct has 1 then the accuracy\_group increases 3 points.
On the other hand, num\_incorrect decreases 1 point when num\_incorrect has 1 and decreases 2 points when num\_incorrect has 2 or more.

This could be expressed as 
```
# num_incorrect[num_incorrect &gt; 2 ] = 2 # Constrained not to exceed 2.
new_accuracy_group = 3 * num_correct - num_incorrect
```

Using the above equation, we can calculate the real values of 0 to 3 from num\_correct, num\_incorrect.
Therefore, the prediction of the model is set to [num\_correct\_pred, num\_incorrect\_pred]

```
prediction = model(x)  # prediction = [num_correct_pred, num_incorrect_pred]
# target = [num_correct; num_incorrect]
```
**Then train the model with the modified_loss below.**
```
modified_loss = mse_loss( prediction, target )
```

After the training is done, we can use the new_accuracy\_group calculated from "num\_correct\_pred, num\_incorrect\_pred".
```
num_correct_pred, num_incorrect_pred = prediction
new_accuracy_group = 3 * num_correct_pred - num_incorrect_pred
```

We can also use original accuracy\_group to slightly improve performance.
```
prediction = model(x)  # prediction = [accuracy_group_pred, num_correct_pred, num_incorrect_pred]
# target = [accuracy_group; num_correct; num_incorrect]
```

**The final\_accuracy\_group is calculated as below.**
```
new_accuracy_group = 3 * num_correct_pred - num_incorrect_pred
final_accuracy_group = (accuracy_group_pred + new_accuracy_group) / 2
```


# Additional training data generation

I generated an additional label for game\_sessions, where the type is **Game**. From the "correct":true, “correct”:false of event\_data, I was able to create num\_correct and num\_incorrect, and likewise I was able to create an accuracy_group.
The number of additional training samples generated is 41,194.

#### Pre-training and fine-tuning steps
- Pre-training step - up to 3 epoch, the model was trained with the original labels + additional labels.
- Fine-tuning step - from 4 epoch, the model was trained with the original labels.


# Data Augmentation
- training time augmentation - For installation\_id with more than 30 game_sessions, up to 50% were randomly removed in the old order.
- test time augmentation - For installation\_id with more than 30 game_sessions, up to 60% were randomly removed in the old order.


## Link to kernel
https://www.kaggle.com/limerobot/dsb2019-v77-tr-dt-aug0-5-3tta?scriptVersionId=27448615

It is a shame to me sharing the uncleane code. But first I decided to share the kernel and make a clean code. Maybe in two weeks? ;)

- The training code is also released. I'm sorry it's still unclean code.
https://github.com/lime-robot/dsb2019
