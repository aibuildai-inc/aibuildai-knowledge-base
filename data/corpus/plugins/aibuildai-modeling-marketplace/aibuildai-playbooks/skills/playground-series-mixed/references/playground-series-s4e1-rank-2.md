# 2nd place solution

Competition: playground-series-s4e1
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s4e1/discussion/472496

Hello everyone. My solution may be a ray of hope for those who don't always have time for many experiments ;)

# The Idea

The main idea (or if you prefer, the only one ;)) behind my solution is to use the 'syntheticity of the dataset' and mostly ignore the topic of the competition.

The first step was to automatically generate all 1,2,3...10 element subsets from the set: ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember' ','EstimatedSalary','Balance'].

Then, for each such subset, checking whether it appears unchanged in the original dataset.
If so, the feature received a value of 1, if not 0.

Example (pseudocode):

```python
subset = ['CreditScore', 'Geography', 'Balance']
x1 = 1 if subset in original_dataset else 0
```

# Why ?

why so? I noticed that the target distribution on the features constructed in this way is completely different than the basic 0.21 for the entire dataset. Using the above feature as an example:

```python
df_train.groupby('x1').Exited.mean() gives the result:
False 0.171026
True 0.269552

or for ['CreditScore', 'Gender', 'NumOfProducts']:
False 0.203034
True 0.707857
```

I also added subset [CustomerId, Surname] to this set of ~1000 features.

There was actually nothing else that needed to be done :)

# Models

The first model was:

**lgbm**: 50 selected features (including basic features), without excessive hyperparameters tuning, on private leaderboard it got 0.90203

**autogluon**: all features included, presets='best_quality', time_limit=14400, on private leaderboard it got  0.90378

**simple ensamble**: (lgbm+autogluon)/2, on private leaderboard it got my final - 0.90462

In the latest submission some of the publicly available features were included, but they didn't change much.

Of course, I also added the @paddykb "Feeling lucky trick" :), in my case it's +0.01 to the result, a little bit less then 0.02, this is probably due to the features that partially explained this 'strange' behavior of the dataset.

and that's it :)

Thank you everyone who has participated in this competition and remember, if you are asked to predict churn in real life, do not ask the client for the original dataset from which yours was generated :)
