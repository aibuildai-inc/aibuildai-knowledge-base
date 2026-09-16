# 21th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #21
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369822

# Overview
* Weighted average ensemble of 21 models.
* Some model used a weighted average of the original prediction and the prediction trained by SVR or Ridge regression of the embedding. (Whether to use SVR or Ridge or neither was based on the pattern that would give the best CV when incorporated into the ensemble)
* cv=0.4407/PublicLB=0.434031/PrivateLB=0.434983


# What worked
* Special tokens for each indicator (Multi Token)
  * A token for each indicator ("[REG_COH]", "[REG_SYN]", etc...) instead of a CLS token was added at the beginning and used.
  * Depending on the backbone, DeBERTa-v3-large performed best this way (CV=0.4478)
* Weighted average with SVR or Ridge regression on a per single model
  * Depends on the model, but can easily be improved by about 0.025 if effective
* AWP
  * Accuracy is about 0.01 to 0.02 better on its own when used
  * However, when incorporated into ensembles, models without AWP often improved performance, so it was ultimately applied to only some models
* Classification Task Model
  * The stand-alone performance was not that good (cv=0.4562), but it was effective for the ensemble.
* Pseudo Labeling
  * Basically all models improved performance (the most effective model improved by about 0.005, cv=0.445)
  * However, PublicLB lost a lot of accuracy and only one pseudo-label model was included in the final sub
  * But this was a huge mistake, and the biggest factor in my shakedown. (See [this discussion](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369401) for more details.)
  * [Reference] Performance of the ensemble model with Pseudo-Label: cv=0.4410/PublicLB=0.436303/PrivateLB=0.433781
* LLRD

# What Didn’t Worked
* Last Layer Re-initialization
* Post Process
  * I tried to correct the final forecast by multiplying the final forecast by the coefficient obtained by Nelder-Mead for each range, but LB worsened, so I did not adopt it.

# Important Citations:
* [tez: fb3 beating the benchmark inference](https://www.kaggle.com/code/abhishek/tez-fb3-beating-the-benchmark-inference)

# Thanks and Acknowledgements:
Thanks to competition organizers for hosting this competition.
And a big thank you to all the participants who shared their wonderful notebooks and discussions.
I was a beginner who first started doing NLP with FB2 about 6 months ago, but thanks to the easy to understand competition design and many useful discussions, I now know a little more about NLP.
I sincerely hope that you will hold such a competition again. Thank you again.
