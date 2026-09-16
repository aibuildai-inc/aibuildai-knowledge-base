# [7th place solution] Essays predictions as features

Competition: linking-writing-processes-to-writing-quality
Rank: #7
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466941

### Acknowledgement
Since public notebooks already have a strong baseline of using action features @awqatak <a href='https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features'>silver bullet notebook</a>, I spend most of my time trying to create features from the annoymized essay using the reconstructed notebook by @jasonheesanglee <a href='https://www.kaggle.com/code/jasonheesanglee/updated-75-35-acc-revealing-hidden-words'>notebook</a> and @kawaiicoderuwu <a href='https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor'>notebook</a>. And special thanks to many many others for their knowledgeable and resourceful notebooks!

<h2>Baseline features</h2>
Select top 100 LGBM feature via feature importance from the 165 features from @awqatak <a href='https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features'>notebook</a>. The validation strategy is 5-fold cross validation (Group-Kfold) which provides a good positive correlation between both CV and LB.

<h3>Essays feature engineering</h3>
- **no_comma**: no of commas "," in the essay
- **no_quotes**: no of quotes " and ' in the essay
- **no_spaces**: no of spaces in the essay
- **no_dot**: no of "." in the essay
- **no_dot_space** = np.log(no_dot) * np.log(no_spaces)

<h3>Generate bag of words using words and paragraphs => 90 features</h3>
- **BOW (words)**: given an essay "qq qqqq qq", the bag of words is [qq: 2, qqqq: 1] for that essay.
- **BOW (paragraphs)**: given an essay of paragraphs "qqq q qq\n\nqq q qq" the bag of words is ["qqq q qq": 1, "qq q qq": 1] for that essay.


<h3>Label preprocessing</h3>
If we look into the distribution of label, the mean value is 3.711. I create another label called "binary label" by encoding the original label into 1 if label >= 3.7 else 0. Also the distribution of the binary label (0: 1224, 1: 1247) is balanced, which is good for training a classification model. A quick sanity test that this binary label works is by leaking this binary label as part of your features, you can get an extremely low CV of 0.4X. Hence if we can predict the encoded label accurately (0 or 1), we can improve our essay prediction by a large margin!

<h3>Essay predictions as feature</h3> 
To use essays as feature we train a simple <a href='https://www.kaggle.com/code/datafan07/train-your-own-tokenizer'>custom BPE tokenizer</a> on the q's to generate BPE tokens which is then use in both CountVectorizer and TfidfVectorizer with ngram_range=(1,3). Finally train two LGBM models on both vectorizers respectively and use their individual binary and regression prediction as another feature.

- **(x2 essay_binary_preds)**
    - Two LGBM trained with Count/Tfidf vectors with binary objective and binary_logloss as metric using encoded/binary label as the target.
    - 1 if probs >= 0.5 else 0
	- The essays CV accuracy is 86%.
- **(x2 essay_regression_preds)**
   - LGBM trained with Count/Tfidf vectors regression objective and rmse as metric using original label as the target.
   - The essays CV rmse is 0.614

<h3>Models</h3>
All models are train with the same features and are tuned using optuna. Final weights are determined by using nelder-mead optimization.
- LGBM (100 features + BOW (words) + BOW (paragraphs) + (x2 essay_binary_preds) + (x2 essay_regression_preds))
- Denselight
- Tabnet

CV: 0.568, public: 0.573, private: 0.562
