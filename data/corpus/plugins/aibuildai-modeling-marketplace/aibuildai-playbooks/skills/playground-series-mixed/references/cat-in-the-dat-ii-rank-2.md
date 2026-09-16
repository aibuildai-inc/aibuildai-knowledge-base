# 2nd-Place-solution [xDeepFM Yes!]

Competition: cat-in-the-dat-ii
Rank: #2
Source: https://www.kaggle.com/c/cat-in-the-dat-ii/discussion/140383

Thanks to the organizers of the Categorical Feature Encoding Challenge II competition, and to the generous sharing and discussion between the kagglers, which helped me a lot.
I will try to describe my entire competition process:   

###Start &amp; early attempts  
When I first participated in this competition, I saw many participants said that simple LogisticRegression works well, so I used the sklearn.linear_model.LogisticRegression model, used hyperopt for hyperparameter search, and then trained the model in 40-fold As a baseline. Later tried catboost, but the effect was not good. I found that in the data set, the discrete classification features accounted for a large proportion, because I was used Wide &amp; deep and DeepFM in other competitions, so I think to use some deep learning for CTR may have better results, and it turned out to be a correct choice.   

### Try CTR   
Part of the training process I used *deepctr* package, source:   
- https://github.com/shenweichen/DeepCTR   
You can install it using **pip install deepctr**   


**FE:** I didn't focus too much on feature engineering, so I just made some normal FE.

```python
data_df['num_null'] = data_df.isna().sum(axis=1)
data_df['ord_5_0'] = data_df['ord_5'].str[0]
data_df['ord_5_1'] = data_df['ord_5'].str[1]
```  

At this stage, I rewritten LogisticRegression(batch_size = 8192, optimizer = 'Adam', regularizers = l2) with TensorFlow as the new baseline.

After that, I tried a variety of CTR models.If there is no special instruction, I use *hyperopt* to perform 5-fold CV on the training set for hyperparameter search, and select the 5 best performing hyperparameter. Finally use 40-fold CV on training set for each hyperparameter and take the average, which is the final score.   

**FM:** FM without linear, score: 0.78661  
**DeepFM:** Use the best 10 hyperparameter combinations, 20-fold cross-validation, score: 0.78922. Thanks to @Siavash wonderful kernel-https://www.kaggle.com/siavrez/deepfm-model, Your idea of using Mish and CyclicLR in training deepfm has greatly influenced me.
**CIN(XdeepFM):** Without Deep and Linear component, score: 0.78940  
**CrossNet(Deep&amp;Cross):** Without Deep and Linear component, score: 0.78888  
**afm:** score: 0.78817  
**autoint:** score: 0.78926   

The reason why I do not use the Deep and linear component in CIN and Cross is because I want to see the performance of these two modules separately, and I want to stacking the models that perform well in the future, and stacking prefers excellent and Different models.   

It can be seen that CIN performs best on the training set. After I submitted these results to the LB, I verified my judgment. But it is interesting that when I stacking all models, although the scores on the training set increased, but the score on LB dropped significantly. Considering the possible risk of overfitting, I abandoned the idea of stacking.    


### Focus on xDeepFM  
- https://arxiv.org/abs/1803.05170
I started to focus on training xdeepfm. By the same method as the period, I got a better score: 0.78955, but the problem followed. we can see that xDeepFM performs best on the training set, but when I put the results to the LB, the score did not rise but fell.

So what should I trust? The cross-validation score on the training set or the Public Leaderboard?   

I carefully analyzed the competition dataset. The training set has 600k rows of data and the test set has 400k rows of data. The PLB only uses 25% of the test data, which is 100k rows. When I use 5-fold cross-validation, the validation dataset for each fold is 120k rows, which is closer to the amount of data used by the PLB, and the validation score between different folds of the same model often jumps between [0.790 and 0.787]. the variance is relatively large. The most important thing is that when using the same kFold iterator (fixed KFold random seed and the number of fold, shuffle=True), the validation scores of different models have correlation, which means that some samples in the data set are difficult to train on any model, even It is impossible to train. According to this, I guess that the reason why the score of the PLB is lower than the k-fold CV is likely to be that the proportion of this part of the "difficult sample" in the PLB is slightly more. The total number of test data rows is similar to the train data, so the final result should be closer to the score of the k-fold CV on training set.  

Finally, I stacking CIN and xdeepfm, and improved the score on training set and PLB. The score in PLB increased to 0.78658(rank 84). But then I believe that the final result will be closer to the score of the train set. It turns out that this guess should correct.
