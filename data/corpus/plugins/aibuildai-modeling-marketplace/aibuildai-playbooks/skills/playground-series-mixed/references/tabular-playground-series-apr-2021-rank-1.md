# 1st place solution - (maybe) swap noise

Competition: tabular-playground-series-apr-2021
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/235739

The result really surprised me! Guess I'm lucky enough to prevent overfitting. I'm sharing one of my [submission notebooks](https://www.kaggle.com/jiangtt/tps-apr-2021-pseudo-labeling-voting-ensemble) but it may not produce the same result because I ran my best submission in my local machine and I blended it with @hiro5299834 's DAE result and @alexryzhkov 's AutoWoe result (remember to change DEBUG to False). This version should get 0.81325 on private LB and get 3rd place.

Actually I'm still not sure if my idea is effective or I'm just lucky to blend the right results as **I don't remember if i turned on swap noise in my best submission**. But I'm sharing it anyways. Maybe you guys can find some luck in it.

From the past 3 tps, I learned from @springmanndaniel that [swap noise](https://www.kaggle.com/springmanndaniel/1st-place-turn-your-data-into-daeta) is key to DAE. After TPS March, I came with an idea to apply swap noise in GBDT training to prevent overfitting, because I think swap noise is a good way to apply data augmentation in tabular data. Unfortunately, no one seems to be interested in my discussion thread, so I decided to try it myself. I modified @ryanzhang 's swap noise function and used it in @hiro5299834 's [amazing work](https://www.kaggle.com/hiro5299834/tps-apr-2021-pseudo-labeling-voting-ensemble). Because I don't know how to apply different noise in every epoch during training, I simply trained many (30 for lgbm, catboost and dt each) models with different noise and mix them.

 I think data augmentation will be useful in GBDT training. If anyone is interested in further research about this idea, please let me know! 

During my training, I learned a lot from automl frameworks. Special thanks to @alexryzhkov for [LightAutoML](https://github.com/sberbank-ai-lab/LightAutoML) and @mt77pp for [MLJAR](https://github.com/mljar/mljar-supervised), both are great automl tools!
