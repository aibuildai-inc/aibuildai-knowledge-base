# 4th Place Solution: Introducing hillclimbers

Competition: playground-series-s3e14
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s3e14/discussion/410639

Hey guys, thank you so much for another fun Playground Series competition! Ever since Season 3 of the PS started in January, I have learned so much and have had such good time participating in these competitions.

I have a lot of people to thank for placing in 4th (please upvote all of these notebooks!):
- @paddykb set the public LB on fire early on in the competition with his excellent notebook [PS s3e14 -FLAML BFI Be-bop-a-blueberry-do-dah](https://www.kaggle.com/code/paddykb/ps-s3e14-flaml-bfi-be-bop-a-blueberry-do-dah) and then once again when he added the [post-processing trick](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/407327) to his model (more on that later).
- @adaubas also set the public LB on fire with another great notebook [PS s3e14 -Stacking - LeastAbsoluteDeviation Reg](https://www.kaggle.com/code/adaubas/ps-s3e14-stacking-leastabsolutedeviation-reg) and discussion post [Some tricks for a 337 score on Public LB](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/409242). This post details how `fruitset`, `seeds`, and `fruitmass` are highly correlated and linearly related to the target and how PCA and PLSRegression can be used for feature engineering along with the code to create the features.
- @francescoliveras shared the highest scoring notebook on the public LB `336.695` [🌟PS-S3-E14🌟 | 📊EDA | Model [EN/ES]](https://www.kaggle.com/code/francescoliveras/ps-s3-e14-eda-model-en-es) which utilizes @paddykb's FLAML AutoML setup and @adaubas' LADRegression blend strategy.
- @tetsutani shared an awesome notebook [PS3E14 EDA| Various models & Ensemble baseline](https://www.kaggle.com/code/tetsutani/ps3e14-eda-various-models-ensemble-baseline) which has good visualizations and where they used a bunch of different models (some models more than once with different hyper parameters) blended together with OptunaWeights to create a high scoring ensemble.
- @chayaphatnicrothanon shared his nice notebook [(LB Score: 338.63) EDA+CatBoost+LightGBM+KFolds](https://www.kaggle.com/code/chayaphatnicrothanon/lb-score-338-63-eda-catboost-lightgbm-kfolds) which has some very cool evaluation based visualizations.
- @zhukovoleksiy shared another great notebook [[PS S3E14] Simple EDA + Ensemble](https://www.kaggle.com/code/zhukovoleksiy/ps-s3e14-simple-eda-ensemble) which uses a similar strategy to @tetsutani.

### The post-processing trick

What did all of the above notebooks have in common? They (and many others) all used the post-processing trick. However, they did not use it all in the same way. Some people used it in their cross validation splits and some people used it after the splits' predictions were blended together. I personally found it beneficial to use it in the cross validation splits **and** after the splits' predictions were blended together.

### What is hillclimbers?

The reason I just showed my gratitude to the all of the above notebooks is because these were all the models I used in my final ensemble! One of the other reasons I did this was to try out my project hillclimbers which is a python module that uses hill climbing to iteratively blend machine learning model predictions. Big shoutout to @cdeotte for his original post explaining and showcasing hill climbing [3rd Place Solution](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609) and @samuelcortinhas for his amazing notebook [📈 PS S3E3 - Hill Climbing like a GM](https://www.kaggle.com/code/samuelcortinhas/ps-s3e3-hill-climbing-like-a-gm). I could not and would not have created hillclimbers without you guys!

### Why did you create hillclimbers?

I created hillclimbers because after competing in a couple of these competitions I quickly realized how important it is to have a diverse model. With hill climbing, the models with the best cross validation scores are not always chosen first. Instead hill climbing chooses diverse models. I started playing around with the code in @samuelcortinhas' notebook and have tried to use it in the past couple Playground competitions. I found myself editing the code a lot from competition to competition and wanted to create something that could be more adaptable. Here are the things I implemented to make it a piece of cake to use hill climbing for almost any tabular problem:

`!pip install hillclimbers`
`from hillclimbers import climb_hill, partial`

```
def climb_hill(
    train=None, 
    oof_pred_df=None, 
    test_pred_df=None, 
    target=None, 
    objective=None, 
    eval_metric=None,
    negative_weights=False, 
    precision=0.01, 
    plot_hill=True, 
    plot_hist=False
) -> np.ndarray: # Returns test predictions resulting from hill climbing
```
- `target`: Let's you specify the target column you are trying to predict.
- `objective`: Set to `"maximize"` or `"minimize"` depending on the evaluation metric you are using.
- `eval_metric`: Define the evaluation metric!
- `negative_weights`: Do you want to use negative weights?
- `precision`: Specifies the step to be taken in the array of weights.

Please visit the [github repo](https://github.com/Matt-OP/hillclimbers) for more detailed explanations.

### Results
***

Now let me show you how hillclimbers works:



And here is the hill climbing plot:



[**You can find all the code for this solution in this notebook**](https://www.kaggle.com/code/mattop/4th-place-hillclimbing-with-5-models-ps-s3-e14)
