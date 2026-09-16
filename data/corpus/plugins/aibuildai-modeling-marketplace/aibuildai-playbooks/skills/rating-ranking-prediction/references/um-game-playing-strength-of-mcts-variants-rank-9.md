# 9th Place Solution: Various Augmentations + A lot of Modeling tricks

Competition: um-game-playing-strength-of-mcts-variants
Rank: #9
Source: https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549624

Thanks to the host and Kaggle for such an amazing experience. I really enjoyed competing in this one especially with my great teammates @cody11null @leehann @gauravbrills.  
And finally after 3 years of competing got a gold medal and proceeded to the Master tier!!!! 


# Summary:
Our solution consists of ensemble between modified AdvantageP1 feature, CatBoost and two LGBM Dart models trained on different data/features.
We managed to chose our 2nd best sub in private which was 2nd in public and one of the highest in cv.

We validated our models using 8-folds GroupKFold. Surprisingly, using 8 folds instead of 5 or 10 made the cv much more stable and correlated more than any other strategy.



# Data Preprocessing:

## 1. Data Augmentation:
We explored several types of relations including the following:
**Flip:  Agent1=a, Agent2=b ==> Agent1=b, Agent2=a**
We flip the agents, change AdvantageP1 to 1-AdvantageP1 and multiply the utility_agent1 by -1.
This is the most useful one. It worked very well for training and as TTA for inference. Especially if you trained a model for original only and a model for augmented samples only then ensembled. This gives huge boost in both cv (+~0.01 cv) and lb (+~0.003 lb) rather than merging them in the same model (+~0.001 lb). 
But at the end, after many experiments we found training on it make the models less stable so dropped it from training and only used original data. Turned out to be right as that our final sub without training on Aug performed better in private.
**Self-play:  Agent1=a, Agent2=b ==> Agent1=a, Agent2=a AND Agent1=b, Agent2=b**
 We create new samples by making the agent against himself, put Advp1 to 0.5 and utility_agent1 to 0. This helped a lot in cv but not in LB.
**Transitivity:  Agent1=a, Agent2=b AND Agent1=b, Agent2=c ==> Agent1=a, Agent2=c**
It was very hard to setup the AdvantageP1 and utility_agent1 values for this one. I adopted this approach:
- Initialize the wins/draws/losses of the new sample to be:
```python
# Assign num_wins_agent1 from first sample
new_row['num_wins_agent1'] = row1['num_wins_agent1']
            
# Assign num_losses_agent1 from second sample
new_row['num_losses_agent1'] = row2['num_losses_agent1']
            
# Sum the draws from both samples
new_row['num_draws_agent1'] = (row1['num_draws_agent1'] + row2['num_draws_agent1']) 
```

- Define "Skill Difference", which refer to the difference in skill between agent 1 and 3, to be: 
```python
diff = row1['num_losses_agent1'] - row2['num_wins_agent1']
```
because if the middle agent in our relation (which connects the two samples) has same wins against the first and 3rd agent, then we can assume agent1 and 3 have no difference in skill, and thus we take the first wins and the the 3rd wins for our new wins and losses. But what if there is a difference? We apply this:
```python
# Calculate the difference in skill
diff = row1['num_losses_agent1'] - row2['num_wins_agent1']
if diff < 0:
      new_row['num_wins_agent1'] += abs(diff)
elif diff > 0:
      new_row['num_losses_agent1'] += abs(diff)
```
We apply the same skill difference idea to the AdvantageP1:
```python
new_row['AdvantageP1'] = row1['AdvantageP1']
new_row['AdvantageP2'] = row2['AdvantageP2']
diff_adv = row1['AdvantageP2'] - row2['AdvantageP1']
clip_val = 1 - (new_row['AdvantageP1'] + new_row['AdvantageP2'])
if diff_adv < 0:
      new_row['AdvantageP1'] += np.clip(abs(diff_adv),0,clip_val)
elif diff_adv > 0:
      new_row['AdvantageP2'] += np.clip(abs(diff_adv),0,clip_val)
```
This actually added whole 200k new samples. Also, with the above modifications the target got distribution very similar to the original training data:

But at the end, there was no significant difference in cv from this transitivity idea, and I didn't try it against LB (I should've done after this overfitting party😅). I think it needs some more work.
Ultimately, we decided to drop all the augmentations ideas from training as they made results much less stable (we used only original data for training), and only use TTA during inference with the flip augmentation. 
For example, we combined the training data and augmented data in X_Train and validated against original data X_Test (to ensure correct comparison) and cv jumped from 0.405 to 0.395. 100% No leakage. But lb went from 0.416 to 0.421. At that point we believed it's easy to overfit on cv, so we tried to use ideas that improve both cv and lb only and that turned out to be the best idea at the end.
Anyway these ideas still might be valid to generate new samples from existing ones but need a bit more improvements I guess.


# 2. Modeling:
I and @cody11null built CatBoost and Dart model based on the above ideas. We didn't use any feature engineering or selection as they didn't help.
@leehann and @gauravbrills built Dart model based on the same ideas but with feature engineering and selection based on correlation, variance and permutation importance for the top 300 features which worked on their dart.
The first CB and Dart was giving cv 0.412 and lb 0.419. The second Dart was giving cv 0.416 and lb 0.421.
Combining both gave us cv 0.406 and lb 0.416.
@leehann and @gauravbrills made some trials with tabnet, xgboost, clustering, embeddings but nothing worked.
For me and @cody11null, we employed several tricks:
- **Subtract AdvantageP1 from the target:**
We are predicting the following:
.png?generation=1733206866485544&alt=media)
AdvantageP1 is very correlated to the target right? So, let's instead, predict df["utility_agent1"] - df["AdvantageP1"] which gives the following distibution:
.png?generation=1733204797074179&alt=media)
Then reverse this at prediction by adding tha AdvP1 again. This actually gives me huge boost in cv ~0.02 and small one in lb <0.000xx.
- **Use AdvantageP1 in the final ensemble:**
AdvantageP1 represents the "Winning probability" right? but what we are predicting is not the winning probability. It is the winning - losing. So, easily replace AdvantageP1 as:
```python
train_df["AdvantageP1"] = (train_df["AdvantageP1"] - (1 - train_df["AdvantageP1"])
```
to make it match how the target is constructed. This gives the following distribution which is similar to the target:
.png?generation=1733205163689424&alt=media)
This didn't help as a feature in the model, but adding it to the final ensemble preds improved cv to 0.404 and lb to 0.415.
- **Predict using Wins/Losses:**
Instead of predicting the utility_agent1 instantly we tried to predict the wins/n and losses/n then subtract them. This also provided huge boost in cv ~0.01 but much worse lb 0.421. I thought that AdvantageP1 might be the reason for this huge boost because (similar to the last point) here we are predicting wins, and AdvantageP1 represents winning probability. So maybe because their meaning and distribution matched then the model relied too much on it? Actually this is where i get the last idea of changing the original advp1 to the difference and it worked at the end, but this wins/losses different models idea didn't work.
- **Using MultiRMSE:**
Use objective="MultiRMSE" in catboost then fit on:
```python
y = pd.concat([y,-y],axis=1)
```
then predict using:
```python
pr = model.predict(test)
pred = (pr[:,0] - pr[:,1]) / 2
```
It gave a boost in cv ~0.002 but slightly worse result in lb 0.416. We chose this as our second sub which gave 0.424 also in private.
- **Model Per Agent:**
Tried to train a model for each agent which gave a boost in cv ~0.002 but same lb 0.416. got worse private 0.425.
- **Full fit:**
I found full fit on the data gave me +0.001 on lb which transitioned to private also.
- **Train a Classifier:**
Reduced the unique values of the target then used a classifier with expectation for the predictions to get to continuous values. Didn't help in cv or lb.
- **Pseudo Labeling:**
Trained my Dart on training data, then continued its training on the ensemble predictions of each 100 samples from test separately and made another new prediction. Didn't help in cv or lb.
- **Optimizing Ensemble weights:**
We optimized for a*(pred)+b for all the models and at the end we optimized for the clip values as well np.clip(pred,a,b). Both was part of our final solution giving boosts in both cv and lb.

You can find our final inference notebook here:
https://www.kaggle.com/code/mohammad2012191/um-inference-nb-9th-place-solution
