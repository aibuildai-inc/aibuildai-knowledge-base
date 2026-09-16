# 4th place solution - preprocess + automl🚀

Competition: playground-series-s4e11
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s4e11/discussion/549197

I'm very happy to have achieved fourth place! Although I believe there is an element of luck involved, I still want to share my solution.

### Preprocessing
After conducting some EDA, I noticed that there was a lot of unreasonable noise in the data. Based on the data preprocessing of @adyiemaz 's [great notebook](https://www.kaggle.com/code/adyiemaz/this-code-fixed-my-depression), I made some modifications, setting unreasonable data to NaN (since the automl framework can automatically handle NaN values). For example, I considered city names like "Less than 5 hours" (which I thought should describe sleep time) as unreasonable. In my experiments, this change resulted in better scores in my CV, public LB, and private LB.

### AutoML
Due to time constraints, I didn't perform very complex model selection. I think AutoML is a great choice to reduce workload while still achieving good results. Ultimately, I used [AutoGluon](https://github.com/autogluon/autogluon) and [LightAutoML](https://github.com/sb-ai-lab/LightAutoML/tree/master), and simply performed an equal-weight blending of their results. Overall, when I finished 8th last time, I observed that blindly blending and overfitting to the public LB led to low private LB scores. In this competition, I am still primarily focusing on the CV score.

#### AutoGluon
AutoGluon provides many pre-tested hyperparameter combinations and bagging+stacking techniques that perform well. Although its CV can sometimes be overly optimistic, it still performed well in this competition. I used the [2024 set of 200 hyperparameters](https://github.com/AutoML-Grandmasters/Fourth-AutoML-Grand-Prix/blob/main/tabrepo_2024_custom.py) provided by the AutoGluon team. Thanks to their work! However, in retrospect, its private LB score was slightly lower than the experimental_quality preset score from version 1.2.0, but it performed better on CV and public LB.

#### LightAutoML
Compared to AutoGluon, LightAutoML offers more deep learning-based models. I used this setup to call all models: `general_params = {"use_algos": [['lgb_tuned', 'cb_tuned', 'mlp_tuned', 'dense_tuned', 'denselight_tuned', 'resnet_tuned', 'snn_tuned', 'node_tuned', 'autoint_tuned', 'fttransformer_tuned']]} `. It performed slightly better than AutoGluon on public and private LB, but slightly worse on CV, which could be due to the overly optimistic CV scores from AutoGluon mentioned earlier.

Finally, I want to thank my college for providing CPU computational support, as well as Kaggle platfrom, all the participants and you, the reader of this article!
