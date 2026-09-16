# [Top 1%] Lessons of this competition... again

Competition: tabular-playground-series-mar-2021
Rank: #11
Source: https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/229834

Hello everybody! 😊
The competition is over. I'm a beginner/contributor in Kaggle and I'd like to share some ideas and lessons.

> First of all, thanks to all participants. Its notebooks and topics are very useful. I learned a lot with them. Some resources are the following:
https://www.kaggle.com/craigmthomas/tps-mar-2021-stacked-starter 
https://www.kaggle.com/hiro5299834/tps-mar-2021-rank-averaging-and-stacking
https://www.kaggle.com/rmiperrier/tps-mar-lgbm-optuna 
https://www.kaggle.com/davidedwards1/tabularmarch21-dae-starter-cv-inference
https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/225929
https://www.kaggle.com/tunguz/tps-mar-2021-eda
https://www.kaggle.com/siavrez/kerasembeddings
https://www.kaggle.com/theawm/0-8917-stratified-kfold-xgboost-eda

In general, i appreciate the rounds of this friendly competitions. I remember my first "serious" model in the january competition: a linear regression 😅😅. Honestly, i'm very happy with my performance and the improvement in my skills. Thanks Kaggle community.

> Anyway, about my solution, like all participants, i used a stacked model. In this moment, my code is dirty.. surely i'll have it in this weekend. It's a promise, well for me. 

At the moment, i mention some general concepts.
**Ensembling:** Lightgbm, Xgboost, Catboost.. these models follows an algorithm. Is a task learn its documentation for optimize it. Also, not forget the other 'single' models as logistic regression, support vector machine, hist gradient boosting, neural networks..
**Stacking:** Combine diverse models. It's logic, if one model fail, other fix that. And only use the best models for each type of model. More information [here](https://mlwave.com/kaggle-ensembling-guide/).
**Overfiting:** The representation of the submission should be general. You can't propose a model without its basics, independiently with the public score, so i recommend use the information of the training to propose predictions.. it sounds obvious but that's the way to avoid the overfit, also define regularizations, see the valid score and use your model in new observations.
**Data leakage:** Some participants upload its notebooks and its score. I think is veery logic to combine your model and the new model. However this way could be incorrect. So, is necessary review its source and evaluate if the new information is good **or** if its addition increases the score or not (an observation here, if the score is high, maybe exist data leakage too, the predictions is lost for example); remember, if you add a model, other model decreases its participation in the model, at least in my ridge on the 2nd stage of modeling.

..Ok, now is turn to learn neural networks, that autoencoders, that theory of unseen and corrupted data.. the tensorflow and keras framework and that all awesome world of deep learning!
