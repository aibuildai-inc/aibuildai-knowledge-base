# 38th Place Solution (3rd Public)

Competition: linking-writing-processes-to-writing-quality
Rank: #36
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466839

Hi everyone,
Thanks for Kaggle and the host for this nice competition! Also, thanks to my brilliant teammates @ihebch and @aliibrahimali. This would be impossible without them!

# Summary:
We used almost same features as the ones in the public notebooks (I think this what led us to shake down in private, maybe because we relied too much on the silver bullet one, but anyway it was a really amazing notebook,  thanks to @awqatak ) and fed them into 3 classifiers (TabPFN, MLPClassifier, MultinomialNB) then took their probabilities as features. Then we fed all the features into our regression ensemble of XGBoost + 6 linear models, then we found the best weights for ensembling using linear regression.

# Validation:
We had a quite good correlation between cv and lb. A lot of the improvements in cv translated to lb. Our best in cv was the best in public and private at the same time. 
Our best in cv was around 0.592, in public lb was 0.570 (3rd public) and in private lb was 0.566. 
Ok what is our cv? It is basically this piece of line:
`StratifiedKFold(n_splits=5/10/.., shuffle=True, random_state=42) `
without any bags (We changed the n_splits several times, but at the end we used 5 because the class of score “0.5” had only 5 elements in the dataset).
We treated the public lb as an additional fold. So, we kept only the changes that improved cv and lb at the same time.

# Feature Engineering:
At the beginning, we started with the features from the public notebooks. We tried adding more aggregations/interactions to them but that didn’t help. What we found to be useful is:
-	Raw Count Features: we added quite a lot of them then did features selection at the end which was useful. (+0.001)
-	Latent Dirichlet Allocation: Using pretrained NLP models didn’t help a lot in our case, so instead of embedding, why not trying some clustering? LDA helped in this and gave us a decent boost in cv/lb. We fed it with count vectorizers with different settings and it was really helpful (+0.002).
-	Some Representation Features: fitting t-SNE on top of the data with different perplexities helped gaining a good boost (+0.001).
-	Probabilities and Expectation of Classifiers: Using raw probabilities and Expectations out of the 3 models: TabPFN, MLPClassifier and MultinomialNB as features gave us a decent boost as well (+0.001). The input for these models was 100 PCA components of the original data. We did that to gain some diversity in input instead of feeding the original data.

# Features Selection:
The abovementioned engineered features were ~1400. We used LOFO extensively several times for all the models. Our final models contain ~350 features only. This gave us another boost (+0.001).

# Modeling:
XGBoost was our best single model. Adding linear models helped gain some diversity.

We didn’t use early stopping as it would lead to overfitting, especially with this small dataset.

# What didn’t work:
-	Training on original logs: We put a lot of time into this one. We tried to fit various types of models on the logs data hoping that we may get some diversity gains later, but nothing worked. Btw, training on these logs by just mapping the target into the IDs won’t be useful (a sample at the beginning of the writing would score same as a sample at the end, which is not correct). So, we decided to use a growing target (Just used np.linspace(0,score,n_samples_in_id)). 
It gave some good boost in its own cv but it still was too bad compared to training on aggregated data (~0.718). It gave some boost in xgboost cv when adding its output as a feature, but that didn’t translate to lb, so we gave it up.
-	More bags: this always gave us a much better cv and much worse lb. So, we gave it up (We are using only ideas that improve both cv and lb). Btw, the result was still bad in private lb.
-	NN: They were overfitting all the time, so we dropped them. The only exception was the MLPClassifier. It gave some boost (as a feature) to the xgboost.
-	Training on Full Data: it looks like diversity matters more here.
-	Pseudo Labeling and Postprocessing: Pseudo Labeling was not working at all. 
Regarding the postprocessing, we wanted to optimize for these a,b,c and d:
```python
preds[preds < a] = b
```
```python
preds[preds > c] = d
```

So, we applied an interior StratifiedKFold inside our cv to find the best values (for the outer cv, we choose n_splits=5 to assure no duplication for the samples of the “0.5” class. For the interior cv, we use n_splits=4 for the same reason). But that wasn’t helpful in the end. I think the low number of samples at the edges was the challenge here.

Here is our code (it is a little messy though):
https://www.kaggle.com/code/ihebch/linking-38th-place-solution-3rd-public?scriptVersionId=158358742

We were just missing that gold to get to the Master tier, but let’s try harder next time.
Thanks for reading!
