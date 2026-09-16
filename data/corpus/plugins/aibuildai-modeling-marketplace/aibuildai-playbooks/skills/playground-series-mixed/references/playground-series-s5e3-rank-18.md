# 18th place solution: single xgboost with custom AUC loss

Competition: playground-series-s5e3
Rank: #18
Source: https://www.kaggle.com/c/playground-series-s5e3/discussion/571021

I’m glad to have maintained a relatively stable ranking in this shake-up competition. In reality, the competition unfolded in two distinct phases for me.
## Early stage
**private score 0.90577**
At this stage, my best model was submitted on March 4th. It was a blending of a neural network and GBDT, based on the following excellent notebooks.
@aryagokh 's [NN notebook](https://www.kaggle.com/code/aryagokh/keras-rainfall/)
@mariusborel 's [GBDT notebook](https://www.kaggle.com/code/mariusborel/rainfall-prediction-stacking/)
The improvements I made can be found in their comments sections. 

And other contributions I made can be found in the forum.

## Later stage
**private score 0.90395**
 In the later stages of the competition, due to my other personal busy tasks and confusion from my overly messy notebooks, I didn't have the time to organize them. So, I decided to abandon my previous efforts, rewrite the notebooks, and spent the last week refining them.

### Dataset
One thing worth noting is that we know about the public dataset, would it be better to incorporate them during training? 
However, in my experiments simulating the private leaderboard, only 50% of the cases achieved a better score. I always feel that I have bad luck, so I gave up on adding them during training.

### Feature
Features are from me,such as [Some cloud values have better predictive accuracy
](https://www.kaggle.com/competitions/playground-series-s5e3/discussion/566054) and from @cdeotte ’s [notebook.](https://www.kaggle.com/code/cdeotte/rapids-svc-w-feature-engineering-lb-0-856)

For me, among various feature selection methods, Recursive Feature Elimination (RFE) consistently delivers better CV results, but its computational cost is prohibitive. Therefore, I adopted the FORWARD FEATURE SELECTION approach outlined in @cdeotte ’s [notebook](https://www.kaggle.com/code/cdeotte/rapids-svc-w-feature-engineering-lb-0-856) for feature selection.

### Kfold
I use `Kfold(6)` (It is equivalent to groupkfold by year) and `Kfold(5,shuffle=True)`.

### Model
xgboost with [AUC custom loss](https://www.kaggle.com/competitions/playground-series-s5e3/discussion/569365)

In fact, someone in an earlier competition used AUC loss and achieved a 4% ranking.
https://www.kaggle.com/code/michaelbryantds/auc-custom-loss-function-top-4

**Why did I choose XGBoost over other GBDT algorithms or the neural networks used by predecessors?**

On the one hand, using custom AUC loss with neural networks is very time-consuming. On the other hand, XGBoost's custom loss functions are more practical. For example, when customizing BCE, XGBoost can produce exactly the same results, while LGBM seems to do some internal optimization, leading to slight differences. Moreover, XGBoost made me aware of the difference in base_score.
 
### Submission seletion

I choose the two notebooks with the highest CV minus public LB.

## Acknowledgements
Thanks  to the authors of the notebooks mentioned above for their significant contributions. Additionally, I would like to express my gratitude to those who actively shared insights in notebooks and forums but were not mentioned. Their dedication and contributions are also greatly appreciated.
