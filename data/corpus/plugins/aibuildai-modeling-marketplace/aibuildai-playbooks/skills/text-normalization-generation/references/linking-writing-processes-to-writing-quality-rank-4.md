# 5th Place Solution: Features are all you need!

Competition: linking-writing-processes-to-writing-quality
Rank: #4
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466961

I'd like to extend my thanks to Kaggle and the Learning Agency Lab for organizing such a fantastic competition. We've had a strong correlation between our cross-validation and leaderboard scores since the beginning, and all three selected submissions are in the gold range, which is quite reassuring.
I would like to thank my teammates. @jaideepvalani @rohitsingh9990 @mori123 @phoenix9032 for their contribution and working hard till the last minute.
 I'll try to briefly summarize what worked and what didn't.

I spent most of my time in feature engineering and the rest on making CV and LB reliable.

# Link to our inference notebook:
https://www.kaggle.com/code/chaudharypriyanshu/light-automl-lgbm-22/notebook

# Feature Engineering

Most of the features are based on the question
**Q.** "what exactly the evaluator is going to see ?"
**A.** The essay text. So the most important features will be derived from it's structure.

So I created features with following categories:

1. Paragraph lengths (first, second, third,.....): word  counts and characters length , some cumulative length features.
2. Sentence lengths (first, second, third): word  counts and characters length 
3. Total capital letters in the text. 
4. total nouns in the text (Total capital letters - total_sentences).
5. Count of words that are added unsequentially.
6. How many sentences start with same word (first word, first two word, first three words).
7. Time window based features : how many words added before (7/15/22/35) minutes.
8. Total commas count.
9. Tfidf features ngram (1,1), A total of 20 features.
10. More Punctuation statistics. Mostly count based.
11. Total questions and exclamation marks. 
12. Total number of more than 1 character replacements. 
13. Cursor positions based features, i used cursor positions that were there in reconstructed essay this way i found what was the actual standard deviation. how many times the author related the cursor with more than 1 positions.
14. Total action time to write words with certain length.
15. Rest of the features were adopted from public notebooks. 

# Modelling
For modelling our team utilised 4 Neural Net, 1 1DCNN based, 3 Trees based models.

###Neural Networks
Most of the neural networks are adopted from lightautoML shared in public notebook [here](https://www.kaggle.com/code/alexryzhkov/lgbm-and-nn-on-sentences).
The architectures we used are:

##### CV LB scores
| Models | cv |LB|
| --- | --- |---|
|  MLP| 0.589 |-|
|  Denselight| 0.590 |-|
|  Autoint| 0.599 |-|
|  NODE| 0.593 |-|
|  1DCNN| 0.602 |0.592|
|  Ensemble| 0.5868 | 0.582|

##### Training Strategy: 
1) Light autoML models are trained for 10 epochs
2) Since the results were a bit unstable, i used Stochastic weighted averaging and used best 3 validation scores (early stopping is used).
3) Since the CV could have been over optimistic i have ensembles them separately from the models that were not using Early stopping.

### Gradient Boosters:

##### CV LB scores 
| Models | cv |LB|
| --- | --- |---|
|  LGBM| 0.598 |0.580|
|  CATBoost| 0.6007 |-|
|  XGBoost| 0.6001 |-|
|  Ensemble| 0.5963 |0.582|

##### Training Strategy: 
1) Trained for 1500 iterations, no early stopping is used.

# Final CV setup
1) Stratified KFold by scores.
2) Trained for 5 seeds [42,2022,7,4,1]
3) Seeds for all models and CV are kept same throught

# Ensembling
1) we independently optimised the weights of NN and Trees and gave equal weights to both.
2) Optuna is used to determine the weight of each model.
Final CV score was 0.5858, LB = 0.578, Private = 0.560

# What we did to avoid shake down?
1. Use same seeds for ensemble
2. Not overfitting on cv, instead we tried to improve both cv and lb and average of both.
3. Added diversity by using various features sets since models trained on different features will deliver diverse results.
4. Separately ensembled the NN(used early stopping) and gradient boosters (do not use early stopping)).
5. Reducing the difference between CV-LB. Most important
6. training all models on same seeds.

As a result
- By adopting this strategy 12 of our submissions out of top 15 reliable and updated solutions were in gold range for us.
- All three final selected subs are under in gold range
- Best CV sub= Best Private LB sub

# What didn't work
1) TFIDF for ngrams more than 1.
2) Past competitions data like Feedback.
3) Word2vec features.
4) svd+tfidf.
5) Timing window based essa structure features.


# Notebooks that helped us throughout the competition, 
**Please give them an upvote**
1) [Feature Engineering: Sentence & paragraph features](https://www.kaggle.com/code/hiarsl/feature-engineering-sentence-paragraph-features)
2) [LGBM and NN on sentences](https://www.kaggle.com/code/alexryzhkov/lgbm-and-nn-on-sentences)
3) [Silver Bullet | Single Model | 165 Features](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features)
4) https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs
