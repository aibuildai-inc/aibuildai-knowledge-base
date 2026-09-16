# 3rd Place Solution - Congratulations New Competition Grandmaster Amed!

Competition: feedback-prize-english-language-learning
Rank: #3
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369609

# Congratulations Grandmaster Amed !
Congratulations Amed ( @amedprof ) for becoming Kaggle's newest Competition Grandmaster! Amed was our teams' superstar who designed most of our models. He truly earns the title Kaggle Competition Grandmaster!

# 3rd Place Solution Summary
Our solution is an ensemble of 24 models without using pseudo labels. Most models use Deberta-v3-large backbone and a variety of tricks. The ensemble was chosen using hill climbing. Teaming with Amed ( @amedprof ) and CroDoc ( @crodoc ) was a pleasure. Thank you guys!

# Hill Climbing - CV 0.4420
I love hill climbing because it can take lots of models and pick the best small subset of models. (i.e. its like Lasso regression) And it computes ensemble model weights. 
  
We start with the single model with best CV score (for us was 0.4470). Then we iterate through all other models and pick a second model which helps the most. Then we pick a third etc until new models do not help. Our ensemble without pseudo achieves CV 0.4420. We allowed hill climbing to use **negative weights** which boosted CV `+0.0010` and private LB `+0.0060` versus using positive weights only. 

    while not STOP:
        potential_new_best_cv_score = GET_BEST()
        for k in range( len(MODELS) ):
            for wgt in range(-0.5,0.51,0.01):
                potential_ensemble = (1-wgt) * current_best_ensemble + wgt * MODELS[k]
                cv_score = compute_metric( potential_ensemble )
                if cv_score < potential_new_best_cv_score: 
                    potential_new_best_cv_score = cv_score
                    REMEMBER_THIS_MODEL(k,wgt)
        current_best_ensemble, STOP = UPDATE_BEST()



# Models - CV 0.4470
Each day, we train new diverse models. Then we run hill climbing to see if the new model gets chosen. We don't need to make a new model with a great CV score, we just need to make new models that are diverse. Below are the choices in order which hill climbing picked from our 50 models. We observe that the best CV score models are not chosen first. Instead hill climbing chooses **diverse models**.

The improvement in ensemble CV score is shown in the plot above. The table below displays single model CV scores. Our best single model without pseudo achieves CV 0.4470

| order selected | backbone  | cv score | weight |
| --- | --- | --- | --- |
|1|deberta-v3-large|0.447|0.190|
|2|deberta-v3-large-squad2|0.4524|0.142|
|3|deberta-v3-large|0.4498|0.124|
|4|deberta-large-mnli|0.4548|0.068|
|5|deberta-v3-large|0.4492|0.133|
|6|xlm-roberta-large|0.4575|0.092|
|7|deberta-v3-large-squad2|0.457|-0.160|
|8|deberta-v3-large-squad2|0.4525|0.101|
|9|deberta-v3-large|0.4489|0.125|
|10|deberta-v3-large-squad2|0.4565|-0.124|
|11|[RAPIDS-SVR][3]|0.4526|0.083|
|12|deberta-v3-large|0.4495|0.102|
|13|deberta-v3-large|0.4502|0.081|
|14|[TF-deberta-v3-base][4]|0.4554|0.061|
|15|deberta-v3-large|0.4527|-0.069|
|16|deberta-v3-large|0.4522|-0.058|
|17|deberta-v3-large|0.45|0.061|
|18|deberta-v3-large|0.4516|-0.058|
|19|deberta-v3-large|0.4509|0.071|
|20|deberta-v3-base|0.4575|-0.039|
|21|deberta-v3-large-squad2|0.4501|0.051|
|22|deberta-v3-large|0.4512|-0.049|
|23|roberta-large|0.4571|0.030|
|24|deberta-v3-large|0.45|0.040|

# What Worked - Tricks
Our models use a diversity of regression heads, pooling techniques, learning schedules, and backbones. Some models used the following tricks:
  
* Different loss rates per target `{'cohesion':0.21, 'syntax':0.16, 'vocabulary':0.10, 'phraseology':0.16, 'grammar':0.21, 'conventions':0.16}` [High Impact] 
* Clip grad norm with `max_norm = 10` [High Impact]
* `hidden_dropout_prob = 0.0` and `attention_probs_dropout_prob = 0.0` [High Impact]
* Stride window for non deberta models [Medium Impact] 
* Since Deberta tokenizer ignores "\n" we replaced "\n\n" with "|" [Medium Impact] 
* Last layer reinitialisation [Medium Impact] 
* 2 stage pooling. First pool either words, sentences, or paragraphs. Then pool that result. [Medium Impact] 
* Train with `max_len=2048`, infer `max_len = 640` [High Impact] 
* Train with `batch_size = 1` [High Impact] 
* RAPIDS SVR using embeddings without train on comp data [Medium Impact] 

# What Didn't Work - Pseudo Labels
Our final submission **does not** use pseudo labels. We tried using pseudo labels during the competition. We were careful to avoid leaks, but none-the-less using pseudo labels from Feedback Prize 1 competition boosted our single model best CV score from 0.4470 to 0.4370 and did not improve our LB score. There must be a leak somewhere but we still haven't found it.

We noticed that the distribution of targets from FP1 is different than FP3. So care must to be taken to account for this difference in distribution. Below are histogram of pseudo labels on FP1 vs. pseudo labels on FP3. We observe that the old Feedback Prize 1 comp has greater target values than current Feedback Prize 3 comp:



# Team Members
* @amedprof
* @crodoc
* @cdeotte 

# Solution Code Published
We published our training code on GitHub [here][1] and we published our inference code on Kaggle [here][2]

[1]: https://github.com/Amed1710/Feedback-Prize--English-Language-Learning
[2]: https://www.kaggle.com/code/cdeotte/3rd-place-solution-lb-0-4337-cv-0-4420
[3]: https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x
[4]: https://www.kaggle.com/code/electro/deberta-layerwiselr-lastlayerreinit-tensorflow
