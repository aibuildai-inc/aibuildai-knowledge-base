# 19th Place Solution

Competition: playground-series-s4e2
Rank: #19
Source: https://www.kaggle.com/c/playground-series-s4e2/discussion/482075

**Notebook :** [link](https://www.kaggle.com/code/samlakhmani/rank-19-91-076-oof-strategy)

**I kept the original data as part of the training set only**
The idea was the in the CV score, I wanted the original data to not influence the accuracy of the CV. Thus I appended the original data. You can observe the model function in the above code. 


**Choosing Folds** 
I experimented with optimization for 10CV 
5CV performed better than 10CV and thus I choose 5CV 
```python
Looks like this remains the only difference between the 19th place and 2nd palace :P 
2nd place used 20CV
```

**Choosing no of time to append Original data to training**
If you look at these two notebooks you will be able to observe experiments I had done with the multiplier. Even My experiments gave 4 as the best multiplier for LGBM, and 1 as a multiplier to the XGB model. 

[Multiplier code]

[Image of accuracy vs no of time data set is appended]

**Thresholding Optimization** 
[This](https://www.kaggle.com/code/samlakhmani/easy-92-196-single-model?scriptVersionId=163207315) is the notebook where I introduced threshold optimization to this competition, post which a lot of people started implementing it. My intention of making it public was to learn on how the implementation can be improved. **It would me a lot to know the impact of the model without the optimization. Please share.** 
Requesting to Upvote the Notebook ^
