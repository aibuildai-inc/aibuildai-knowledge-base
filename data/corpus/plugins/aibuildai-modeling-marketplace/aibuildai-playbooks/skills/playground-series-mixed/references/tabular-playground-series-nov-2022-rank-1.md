# 1st Place Solution

Competition: tabular-playground-series-nov-2022
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-nov-2022/discussion/369674

Hey guys, Thanks for all the competition we had a great time dealing with this dataset.
I’m going to follow @alexryzhkov and explain a bit of our team’s solution.

### Team name

🌪️, @jcaliz and @samuelcortinhas 

### Team place
7 public LB and 1st private LB

### The brief description of the solution

Lots of ensembles with different models architectures and different features. Final blend consists of:
1. LightAutoML provided by @sergiosaharovskiy, @alexryzhkov
2. One XGBoost models trained with different features.
3. One LGBM models trained with different features.
4. Two dense NN trained with different features.

### Used libraries/algorithms:

We used: XGBoost, LightGBM, LightAutoML, Keras, and Weights and Biases among others.

### The key points which improve your score the most

1. Good cross validation, we used 10 StratifiedKfold and other models with 20 StratifiedKfold
2. Stacking using scipy.minimize thanks @pourchot 
3. Multistart the NNs to mitigate noise.
4. Post processing by adding little values to predictions above and below 0.5. Always trusting the CV to select the positive and negative delta. Earlier in the competition EDA notebooks shown that models had more troubles to classify negative labels.
5. Blend different types of models, not only on on the algorithm but o  the features. We selected feature using Top-K AUC, L1 with o LogisticRegression, Feature importance (similar to RFE), permutation importance.
6. Use logits instead of probablities, thanks @ambrosm 
7. Trust your CV its easy to overfit
8. Swish and Mish for activation functions.

### Several ideas that didn't work

1. Boltzmann ensemble, we didn’t go deeper after the first couple of iterations
2. Geometric mean, similar to Boltzman.
3. Residual blocks (a simple neural network was enough)

### What can be done in a better way?

From my experience this is strange competition, because there were almost zero data analysis, I felt it was more like shoot everything until it works. Speaking from myself, I spent too much time trying to do things from scratch instead of reading others work.

### Lessons learned from the competition

There is a huge importance of using an experiment tracking tool, it helps a lot to save oof preds and validate your hypothesis. Be really open to learn new things as required, AutoML is a great example.

Special thanks to my teammate, I enjoyed a lot working in a team, and there is no better way to learn than having a common enemy haha.

If you find this post helpful, make sure to upvote the notebooks that provided insigths:
*  [https://www.kaggle.com/code/mikhailkuz/lightautoml-nn-happiness](https://www.kaggle.com/code/mikhailkuz/lightautoml-nn-happiness)
* [https://www.kaggle.com/code/alexryzhkov/5k-features-not-a-problem-for-lightautoml](https://www.kaggle.com/code/alexryzhkov/5k-features-not-a-problem-for-lightautoml)
* [https://www.kaggle.com/code/ambrosm/tpsnov22-eda-which-makes-sense](https://www.kaggle.com/code/ambrosm/tpsnov22-eda-which-makes-sense)
* [https://www.kaggle.com/code/pourchot/stacking-with-scipy-minimize](https://www.kaggle.com/code/pourchot/stacking-with-scipy-minimize)
* [https://www.kaggle.com/code/sergiosaharovskiy/tps-nov-2022-in-automl-we-trust](https://www.kaggle.com/code/sergiosaharovskiy/tps-nov-2022-in-automl-we-trust)
