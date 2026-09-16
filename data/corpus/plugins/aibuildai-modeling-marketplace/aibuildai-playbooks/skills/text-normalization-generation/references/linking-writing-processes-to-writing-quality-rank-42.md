# Place 46 Solution, Place 11 Efficiency

Competition: linking-writing-processes-to-writing-quality
Rank: #42
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466774

Edit 1: Place in efficiency track -> 11

#Preface
I'm really happy that I survived the shakeup and look forward to also see the results for the efficiency track!
 
I first want to thank everyone involved in this competition, especially the collaborative spirit of the community is always great to see! In the competition I've focused both on the efficiency and performance track, as I’ve had good experience with this approach in a previous one, so I’ll discuss both in one write-up. 

#Solutions
**Performance Track Place 46:**
My final submission was mostly an ensemble of public notebooks. One addition I introduced was the transformer Deberta-v3-xsmall on reconstructed essays. The ensemble of public notebooks got me to 0.577 Public LB, the transformer got me to 0.576 LB and improved CV a lot.

*Ensemble*
The ensemble included a blend of models from public notebooks and the transformer: [Voting Regressor](https://www.kaggle.com/code/seoyunje/gan-with-tabular-data), [Fusion](https://www.kaggle.com/code/kononenko/lgbm-x2-nn-fusion), [GAN](https://www.kaggle.com/code/seoyunje/gan-with-tabular-data), and a Transformer. The final prediction was a weighted average of these models, carefully balanced to optimize performance.

`
y_pred_ensemble = vote_pred * 0.45 + gan_pred * 0.05 + transformer_pred * 0.15 + automl_pred * 0.05 + biglgb_pred * 0.05 + hugelgb_pred * 0.15 + fusion_pred * 0.1
`

*Transformer*
The approach here was to create essays, find common anonymized words and add them to a tokenizer, then train the model. I froze the first 6 of 12 layers of Deberta-v3-xsmall, but not the token embeddings, as the new tokens do not have useful embeddings at initialization. The implementation was inspired by a [popular notebook from a previous competition](https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-spell-autocorrect). I only trained the model on the few texts given for the competition, but I think one might have improved this a lot by pretraining the q embeddings on different anonymized texts (not done due to time constraints and it would also miss the point of the competition a bit to be fair).

```python
important_anon_words = {word: count for word, count in word_counts.items() if count > 100} 
important_anon_words = {k: v for k, v in important_anon_words.items() if len(k) > 1} 
tokenizer.add_tokens(list(important_anon_words))
inputs = tokenizer("qqqq qq qqq?", return_tensors="pt")

# Before adding tokens
print("'input_ids':", tensor([[ 101, 1053, 4160, 4160, 4160, 1053, 4160, 1053, 4160, 4160, 1029, 102]]))

# After adding tokens
print("'input_ids':", tensor([[ 101, 30525, 30523, 30550, 102]]))
```

**Efficiency Track Place 11:**
Here I deployed a 5 fold x 5 model ensemble using CatBoost, XGB, LGBM, linear SVR, Ridge. Pretty similar to the Voting Regressor in the end with some differences in features and model selection. It runs in under 2 minutes.

*Timing*
A significant amount of time was taken up by feature loading, with a focus on optimizing the selection of features for better efficiency. I used the [Silver Bullet features](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features), throwing out less relevant features based on LGB feature importance across CV folds. Only a few seconds in inference are used on model running. 

I noticed that this time every second counts, based on observed differences in efficiency LB. I think that they took something like a 0 submission for base calculation which would make the difference between 580 and 579, if minimum score is 570:

First part of Efficiency equation: (580-579)/(3700-560) => 0.001 point on LB is worth around 1/3140 score in this case, same as around 10 seconds runtime based on second part of Efficiency equation!
I've got 571 on private test with my 2 minute submission so really hope it will be enough for a good place!


*Cross Validation*
I ended up using a 5 fold CV on half of data and then used the other half as test set (switching around). This way you can use early stopping and see performance gains from ensembling over folds, while still seeing performance on an unbiased set, when training the final models for inference you can then switch to using all data for training.


*Post Processing and Distribution Shift*
Some things improved CV by a lot, like the addition of ridge or linear SVR to an ensemble of gradient boosting models, but did not translate to great improvements on LB. 

I think the reason might be a distribution shift to test set, which is probably harder to handle for regression models. They are not automatically capped to one range by default. You can see them predicting negative numbers on CV sometimes for example. I tried post processing with both tracks to alleviate some of these issues, but nothing worked really well sadly. I hope this will clear up once more write-ups appear!

#Takeaways
Good cross validation is really important, I tapped in the dark a lot in the beginning.
Teaming up to cover more ground would have been good, I saw myself not able to pursue lots of good avenues, I hoped I could get solo gold for GM as I was really optimistic but that was probably a missed team gold instead.
